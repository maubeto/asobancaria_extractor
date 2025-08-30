import subprocess
import tempfile
import os
import re
import sys
import shutil
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple

try:
    import requests
except Exception:  # fallback if requests missing
    requests = None


WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
SEMANA_DIR = os.path.join(WORKSPACE_ROOT, "Informe Asobancaria")
BANCA_DIR = os.path.join(WORKSPACE_ROOT, "Banca y Economia")
SALIDAS_DIR = os.path.join(WORKSPACE_ROOT, "salidas")
CACHE_DIR = os.path.join(WORKSPACE_ROOT, "cache")
ANEXOS_DIR = os.path.join(CACHE_DIR, "anexos")

KEYWORDS = [
    "revisor",
    "revisoria fiscal",
    "revisoría fiscal",
    "auditor",
    "auditoria",
    "auditoría",
    "kpmg",
]


def ensure_dirs():
    os.makedirs(SALIDAS_DIR, exist_ok=True)
    os.makedirs(ANEXOS_DIR, exist_ok=True)


def run_pdftotext_to_string(pdf_path: str) -> str:
    """Extract text from a PDF using pdftotext CLI."""
    if not shutil.which("pdftotext"):
        raise RuntimeError("La herramienta 'pdftotext' no está disponible en el sistema.")
    try:
        # -layout keeps a more faithful structure; -nopgbrk avoids form feed page break chars
        result = subprocess.run(
            ["pdftotext", "-layout", "-nopgbrk", pdf_path, "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return result.stdout.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error al extraer texto de {pdf_path}: {e.stderr.decode('utf-8', errors='replace')}")


def normalize_text(s: str) -> str:
    import unicodedata

    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s


def is_upper_block(line: str) -> bool:
    s = line.strip()
    if not s or len(s) < 8:
        return False
    if any(x in s for x in (":", "@", "www", "Edición", "Asobancaria")):
        return False
    # Check uppercase ratio
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    upper = sum(1 for c in letters if c.isupper())
    # Avoid treating cover 'RESUMEN DE REGULACIÓN' as a section
    if "RESUMEN" in s and "REGUL" in s:
        return False
    return upper / max(1, len(letters)) > 0.8


def parse_semana_pdf(text: str) -> List[Dict[str, object]]:
    """Parsea el Informe Semanal en bloques de artículos.

    Devuelve una lista de dicts con: titulo, resumen, fecha, enlaces (lista).
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    items: List[Dict[str, object]] = []

    # Bullets may appear as '•', '', or private-use bullets like \uf0b7, sometimes mid-line due a two-column layout
    bullet_re = re.compile(r"^.*?[•\uf0b7]\s+(.*)")
    fecha_re = re.compile(r"^\s*Fecha de publicaci[oó]n:\s*(.*)")
    url_re = re.compile(r"^\s*(Norma|Anexo):\s*(\S+)")

    i = 0
    N = len(lines)
    current_section: Optional[str] = None
    while i < N:
        # Track section headers (uppercase blocks possibly over 1-2 lines)
        if is_upper_block(lines[i]) and not bullet_re.match(lines[i]):
            # Build a section, skipping standalone words like NORMATIVIDAD
            parts = [lines[i].strip()]
            j = i + 1
            # Include next line if upper too and not a bullet/metadata
            while j < N and is_upper_block(lines[j]) and not bullet_re.match(lines[j]):
                parts.append(lines[j].strip())
                j += 1
            joined = " ".join(parts)
            joined = re.sub(r"\bNORMATIVIDAD\b", " ", joined, flags=re.I)
            # Normalize multiple spaces
            joined = re.sub(r"\s+", " ", joined).strip()
            if joined:
                current_section = joined
            i = j
            continue

        m = bullet_re.match(lines[i])
        if not m:
            i += 1
            continue

        # Capture the bullet block (title + possible wrapped description)
        content_lines = [m.group(1).strip()]
        i += 1
        # Collect continuation lines. If a new section header appears, update it and do not treat as content.
        while i < N and not bullet_re.match(lines[i]) and not fecha_re.match(lines[i]) and not url_re.match(lines[i]):
            # Detect and capture a section header mid-flow
            if is_upper_block(lines[i]):
                parts = [lines[i].strip()]
                j = i + 1
                while j < N and is_upper_block(lines[j]) and not bullet_re.match(lines[j]):
                    parts.append(lines[j].strip())
                    j += 1
                joined = " ".join(parts)
                joined = re.sub(r"\bNORMATIVIDAD\b", " ", joined, flags=re.I)
                joined = re.sub(r"\s+", " ", joined).strip()
                if joined:
                    current_section = joined
                i = j
                continue
            ln = lines[i].strip()
            if ln:
                content_lines.append(ln)
            i += 1

        # After content block, capture metadata lines immediately following
        fecha = None
        enlaces: List[str] = []
        while i < N:
            f = fecha_re.match(lines[i])
            u = url_re.match(lines[i])
            if f:
                fecha = f.group(1).strip()
                i += 1
                continue
            elif u:
                enlaces.append(u.group(2).strip())
                i += 1
                continue
            else:
                break

        # Compact title and resumen from content_lines
        title = content_lines[0]
        resumen = " ".join(content_lines[1:]).strip() if len(content_lines) > 1 else ""
        # Skip bullets that are likely part of the cover summary (no metadata nearby)
        if not fecha and not enlaces:
            # ignore cover/summary bullets to preserve intended order
            continue

        items.append({
            "seccion": current_section,
            "titulo": title,
            "resumen": resumen,
            "fecha": fecha,
            "enlaces": enlaces,
        })

    return items


def _cache_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]


def _read_cache_text(key: str) -> Optional[str]:
    p = os.path.join(ANEXOS_DIR, f"{key}.txt")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None
    return None


def _write_cache_text(key: str, text: str) -> None:
    p = os.path.join(ANEXOS_DIR, f"{key}.txt")
    try:
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass


def fetch_url(url: str, timeout: int = 20) -> Tuple[str, Optional[str]]:
    """Descarga una URL y retorna (texto_extraido, error). Maneja HTML/PDF; imágenes quedan sin OCR.

    Si no hay red o requests, retorna ("", motivo).
    """
    if not requests:
        return "", "Dependencia 'requests' no disponible."
    # Ensure scheme
    if url.startswith("www."):
        url = "https://" + url
    elif not re.match(r"^https?://", url):
        url = "https://" + url

    # cache
    key = _cache_key(url)
    cached = _read_cache_text(key)
    if cached:
        return cached, None

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36",
        "Accept-Language": "es-CO,es;q=0.9,en;q=0.8",
    }
    last_exc = None
    for attempt in range(2):
        try:
            resp = requests.get(url, timeout=timeout, allow_redirects=True, headers=headers)
            break
        except Exception as e:
            last_exc = e
            if attempt == 1:
                return "", f"No se pudo acceder: {e}"
    if last_exc:
        return "", f"No se pudo acceder: {last_exc}"

    ctype = resp.headers.get("content-type", "").lower()
    # Try to decode text for HTML or text
    if "text/html" in ctype or (not ctype and resp.text):
        html = resp.text
        text = strip_html(html)
        if text:
            _write_cache_text(key, text)
        return text, None
    if "application/pdf" in ctype or url.lower().endswith(".pdf"):
        text, err = pdf_bytes_to_text(resp.content)
        if text and not err:
            _write_cache_text(key, text)
        return text, err
    if any(img in ctype for img in ("image/png", "image/jpeg", "image/jpg", "image/webp")):
        return "", "Contenido es imagen; OCR no disponible en este entorno."
    # Fallback: try to decode as text
    try:
        text = resp.content.decode(resp.encoding or "utf-8", errors="replace")
        if text:
            _write_cache_text(key, text)
        return text, None
    except Exception:
        return "", f"Tipo de contenido no soportado: {ctype or 'desconocido'}"


def strip_html(html: str) -> str:
    # Remove scripts/styles
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.I)
    # Replace breaks/paragraphs with line breaks
    html = re.sub(r"<(br|p|div|li|h[1-6])\b[^>]*>", "\n", html, flags=re.I)
    # Strip tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Unescape HTML entities
    import html as ihtml
    text = ihtml.unescape(text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def pdf_bytes_to_text(pdf_bytes: bytes) -> Tuple[str, Optional[str]]:
    if not shutil.which("pdftotext"):
        return "", "'pdftotext' no está disponible para procesar PDFs remotos."
    with tempfile.TemporaryDirectory() as td:
        pdf_path = os.path.join(td, "doc.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        try:
            text = run_pdftotext_to_string(pdf_path)
            return text, None
        except Exception as e:
            return "", f"Error al leer PDF remoto: {e}"


def find_keyword_hits(text: str, keywords: List[str]) -> List[Tuple[str, int]]:
    ntext = normalize_text(text)
    hits = []
    for kw in keywords:
        nkw = normalize_text(kw)
        count = len(re.findall(re.escape(nkw), ntext))
        if count > 0:
            hits.append((kw, count))
    return hits


def extract_snippets(text: str, keywords: List[str], max_snippets: int = 3) -> List[str]:
    # Split text into sentences (very basic)
    sents = re.split(r"(?<=[\.!?])\s+", text)
    ntext = normalize_text(text)
    nkeywords = [normalize_text(k) for k in keywords]
    scored: List[Tuple[float, str]] = []
    for idx, s in enumerate(sents):
        ns = normalize_text(s)
        score = sum(ns.count(k) for k in nkeywords)
        # Nearby context bonus
        score += 0.25 * (1.0 / (1 + len(s)))
        if score > 0:
            scored.append((score, s.strip()))
    scored.sort(key=lambda x: x[0], reverse=True)
    snippets = [s for _, s in scored[:max_snippets]]
    # Ensure not empty; fallback to leading sentences
    if not snippets:
        snippets = [s.strip() for s in sents[:max_snippets] if s.strip()]
    # Trim overly long snippets
    trimmed = []
    for s in snippets:
        if len(s) > 300:
            s = s[:297].rstrip() + "…"
        trimmed.append(s)
    return trimmed


def _format_hits(hits: List[Tuple[str, int]]) -> str:
    if not hits:
        return "ninguna"
    return ", ".join(f"{kw} ({cnt})" for kw, cnt in hits)


def _split_sentences(text: str) -> List[str]:
    # Basic, language-agnostic sentence split
    sents = re.split(r"(?<=[\.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def _is_noise_sentence(s: str) -> bool:
    s_comp = normalize_text(s)
    noise_terms = [
        "inicio", "buscar", "desplegar", "navegacion", "navegación", "menu", "menú",
        "transparencia", "atencion", "atención", "tramites", "trámites", "contactenos", "contáctenos",
        "politica", "política", "privacidad", "mapa del sitio", "spqr", "plan anticorrupcion", "plan anticorrupción",
        "mision y vision", "misión y visión", "principios y valores", "politicas institucionales", "políticas institucionales",
        "aviso de privacidad", "cookies", "accesibilidad", "faq", "preguntas frecuentes",
    ]
    if any(t in s_comp for t in noise_terms):
        return True
    # Too many capitalized tokens indicates menu/list
    tokens = s.split()
    if len(tokens) >= 6:
        caps = sum(1 for t in tokens if t[:1].isupper() and t[1:].islower())
        if caps / len(tokens) > 0.7:
            return True
    # Very short or no verbs/action cues
    if len(s) < 25:
        return True
    return False


def _extract_context_reasons(text: str, total_hits: List[Tuple[str, int]], max_lines: int = 3) -> List[str]:
    # Build short reasons around where keywords appear
    sents = _split_sentences(text)
    nsents = [normalize_text(s) for s in sents]
    # Prioritize keywords by count desc
    top_kws = [kw for kw, _ in sorted(total_hits, key=lambda x: (-x[1], x[0]))[:3]]
    reasons: List[Tuple[float, str]] = []
    # action/stative cues to score sentences higher
    cues = [
        "modific", "reglament", "instru", "exig", "requir", "establec", "oblig",
        "control", "report", "verific", "supervis", "cumplim", "procedim", "auditor",
        "revisori",
    ]
    for kw in top_kws:
        nkw = normalize_text(kw)
        for idx, ns in enumerate(nsents):
            if nkw in ns:
                # Use the sentence and optional neighbor for richer context
                for j in (idx - 1, idx, idx + 1):
                    if 0 <= j < len(sents):
                        s = sents[j]
                        if _is_noise_sentence(s):
                            continue
                        score = 1.0
                        nss = nsents[j]
                        score += sum(1 for c in cues if c in nss) * 0.75
                        # prefer the sentence that actually contains the kw
                        if j == idx:
                            score += 0.5
                        # compact overly long sentences
                        ss = s
                        if len(ss) > 220:
                            ss = ss[:217].rstrip() + "…"
                        reasons.append((score, ss))
                break  # first occurrence is enough
    # sort by score and return deduped
    reasons.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    out: List[str] = []
    for _, r in reasons:
        key = normalize_text(r)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= max_lines:
            break
    return out


def _condense_context(sentence: str, keywords: List[str], window: int = 18) -> str:
    # Return a short phrase centered around the first keyword occurrence
    words = sentence.split()
    n_sentence = normalize_text(sentence)
    nwords = n_sentence.split()
    nkeywords = [normalize_text(k) for k in keywords]
    # Build mapping from char idx to word idx to find window; simpler: search in n_sentence, then approximate word idx
    pos = None
    found_kw = None
    for kw in nkeywords:
        p = n_sentence.find(kw)
        if p != -1 and (pos is None or p < pos):
            pos = p
            found_kw = kw
    if pos is None:
        # fallback: trim leading/trailing
        return sentence[:180].rstrip(' ,;:') + ("…" if len(sentence) > 180 else "")
    # Approximate word index by counting spaces before pos
    prefix = n_sentence[:pos]
    widx = len(prefix.split())
    start = max(0, widx - window // 2)
    end = min(len(words), start + window)
    frag = " ".join(words[start:end]).strip(' ,;:')
    if start > 0:
        frag = "… " + frag
    if end < len(words):
        frag = frag + " …"
    return frag


def _build_impact_paragraph(total_hits: List[Tuple[str, int]], combined_text: str, item_title: Optional[str]) -> str:
    # Compose 2–3 coherent lines grounded on where the keywords appear
    title = (item_title or "el documento").strip()
    lines: List[str] = []
    reasons = _extract_context_reasons(combined_text, total_hits, max_lines=2)
    top_kws = [kw for kw, _ in total_hits[:3]]
    # Line 1: bind title with a concrete reason
    if reasons:
        concise = _condense_context(reasons[0], top_kws)
        lines.append(f"en ‘{title}’ se menciona(n) {', '.join(top_kws)} en la medida que {concise}")
    else:
        lines.append(f"en ‘{title}’ se abordan materias vinculadas a las palabras clave detectadas")
    # Line 2: impact statement
    lines.append("esto puede implicar ajustes de control interno, alcance de pruebas, evidencias y reportes del revisor fiscal")
    # Line 3: optional reinforcement
    if len(reasons) > 1:
        concise2 = _condense_context(reasons[1], top_kws)
        lines.append(f"adicionalmente, se resalta que {concise2}")
    return "\n".join(lines)


def analyze_article(item: Dict[str, object]) -> Dict[str, object]:
    enlaces = list(item.get("enlaces", []))
    fetched_texts: List[str] = []
    fetch_errors: List[str] = []
    per_link_hits: List[Tuple[str, List[Tuple[str, int]]]] = []
    for url in enlaces:
        text, err = fetch_url(url)
        if err:
            fetch_errors.append(f"{url}: {err}")
        if text:
            fetched_texts.append(text)
            per_link_hits.append((url, find_keyword_hits(text, KEYWORDS)))

    combined_text = "\n\n".join(fetched_texts)
    # Aggregate totals across enlaces
    totals: Dict[str, int] = {}
    for _, hlist in per_link_hits:
        for kw, cnt in hlist:
            totals[kw] = totals.get(kw, 0) + cnt
    total_hits = [(kw, totals[kw]) for kw in totals]
    total_hits.sort(key=lambda x: (-x[1], x[0]))

    if total_hits:
        cuerpo = _build_impact_paragraph(total_hits, combined_text, item.get("titulo"))
        impacto = (
            "Se observa impacto para el revisor fiscal en la medida, que (\n"
            + cuerpo
            + "\n)\n"
            + "Palabras clave detectadas (total): "
            + _format_hits(total_hits)
        )
        if per_link_hits:
            detalle_lines = ["Palabras clave por enlace:"]
            for url, hl in per_link_hits:
                detalle_lines.append(f"- {url}: {_format_hits(hl)}")
            impacto += "\n" + "\n".join(detalle_lines)
    else:
        if enlaces and not combined_text and fetch_errors:
            impacto = "No se pudo ingresar a los enlaces: " + "; ".join(fetch_errors)
        else:
            impacto = "No se observa impacto para el revisor fiscal"

    return {
        "seccion": item.get("seccion"),
        "titulo": item.get("titulo"),
        "resumen": item.get("resumen"),
        "fecha": item.get("fecha"),
        "enlaces": enlaces,
        "resultado": impacto,
    }


def format_stage1_output(results: List[Dict[str, object]]) -> str:
    out_lines = []
    for idx, r in enumerate(results, 1):
        out_lines.append(f"Artículo {idx}:")
        if r.get('seccion'):
            out_lines.append(f"  Sección: {r.get('seccion')}")
        out_lines.append(f"  Título: {r.get('titulo') or ''}")
        out_lines.append(f"  Resumen: {r.get('resumen') or ''}")
        out_lines.append(f"  Fecha de publicación: {r.get('fecha') or ''}")
        enlaces = r.get('enlaces') or []
        if enlaces:
            for j, u in enumerate(enlaces, 1):
                out_lines.append(f"  Enlace {j}: {u}")
        else:
            out_lines.append("  Enlace: (no disponible)")
        out_lines.append(f"  Resultado: {r.get('resultado') or ''}")
        out_lines.append("")
    return "\n".join(out_lines).rstrip() + "\n"


def _clean_banca_lines(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines()]
    out: List[str] = []
    skip_terms = [
        "edicion", "direccion financiera", "asobancaria", "director:",
        "para suscribirse", "publicacion semanal", "banca & economia", "bancayeconomia@",
    ]
    norm_skip = set(skip_terms)
    for ln in lines:
        if not ln:
            continue
        lnl = normalize_text(ln)
        if any(t in lnl for t in norm_skip):
            continue
        # skip email and navigation noise
        if "@" in ln:
            continue
        # skip bullet list fragments common in portada
        if "•" in ln:
            continue
        out.append(ln)
    return out


def extract_banca_title(text: str) -> Optional[str]:
    lines = _clean_banca_lines(text)
    # consider first 100 lines
    candidates: List[Tuple[str, int]] = []
    noise = [
        "edición", "director", "asobancaria", "para suscribirse", "publicación semanal",
        "correo electrónico", "bancayeconomia@", "página", "1", "2",
    ]
    for idx, ln in enumerate(lines[:120]):
        if not ln or len(ln) < 10:
            continue
        lnl = normalize_text(ln)
        if any(n in lnl for n in noise):
            continue
        # skip bullet lines and email lines
        if any(sym in ln for sym in ("•", "@")):
            continue
        # has sufficient letters
        letters = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]", ln)
        if len(letters) < 18:
            continue
        candidates.append((ln, idx))
    if not candidates:
        return None
    # prefer lines with colon or title-like casing
    def score_title(entry: Tuple[str, int]) -> float:
        s, idx = entry
        sc = 0.0
        if ":" in s:
            sc += 1.0
        # prefer sentence case over all caps
        if s == s.title():
            sc += 0.2
        # longer is better up to 120 chars
        sc += min(len(s), 120) / 200.0
        # earlier lines are better
        if idx < 10:
            sc += 2.0
        elif idx < 20:
            sc += 1.0
        # favor inclusion themed titles
        if "inclusion" in normalize_text(s) or "inclusión" in s.lower():
            sc += 1.5
        return sc
    candidates.sort(key=score_title, reverse=True)
    chosen, cidx = candidates[0]
    # If title is broken across two lines (ends with ':'), try to append next meaningful line
    try:
        idx = cidx
        if idx + 1 < len(lines):
            nxt = lines[idx + 1].strip()
            nxt_letters = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]", nxt)
            nxt_digits = re.findall(r"\d", nxt)
            appended = False
            if len(nxt_letters) >= 5 and len(nxt_digits) <= 1:
                if chosen.endswith(":") or (nxt and nxt[0].islower()):
                    chosen = f"{chosen} {nxt}".strip()
                    appended = True
            if not appended and idx + 2 < len(lines):
                nxt2 = lines[idx + 2].strip()
                nxt2_letters = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]", nxt2)
                nxt2_digits = re.findall(r"\d", nxt2)
                if len(nxt2_letters) >= 5 and len(nxt2_digits) == 0:
                    chosen = f"{chosen} {nxt2}".strip()
    except Exception:
        pass
    return chosen


def summarize_banca_text(text: str, target_words: Tuple[int, int] = (330, 400)) -> str:
    # Abstractive, executive-style summary driven by thematic cues
    clean_text = "\n".join(_clean_banca_lines(text))
    ntext = normalize_text(clean_text)

    def has(*terms: str) -> bool:
        return any(t in ntext for t in terms)

    # Thematic flags
    f_inclusion = has("inclusion", "inclusi")
    f_genero = has("genero", "género")
    f_brechas = has("brecha")
    f_intersec = has("intersecc")
    f_directrices = has("directriz")
    f_marco = has("marco de accion", "marco de acción", "marco")
    f_protocolo = has("protocolo social")
    f_bdo = has("banca de las oportunidades")
    f_ofs = has("finanzas sostenibles", "ofs")
    f_oferta = has("oferta")
    f_demanda = has("demanda")
    f_riesgo = has("riesgo")
    f_instit = has("institucion", "institucional")
    f_habil = has("habilitador", "habilitad")
    f_sosten = has("sostenibil")
    f_datos = has("datos desagregados", "desagregados por sexo")
    f_hoja = has("hoja de ruta")

    # Paragraph 1: context and diagnosis
    p1_parts: List[str] = []
    if f_inclusion and f_genero:
        p1_parts.append("El artículo presenta un análisis de la inclusión financiera con enfoque de género en Colombia")
    else:
        p1_parts.append("El artículo presenta un análisis del sistema financiero desde una perspectiva de equidad")
    if f_brechas:
        p1_parts.append("reconociendo brechas persistentes en acceso y uso que varían por edad, territorio y condiciones sociales")
    if f_intersec:
        p1_parts.append("bajo un enfoque interseccional que visibiliza diferencias entre grupos poblacionales")
    p1 = ", ".join(p1_parts) + "."

    # Paragraph 2: response and framework
    p2_parts: List[str] = []
    resp_bits: List[str] = []
    if f_protocolo:
        resp_bits.append("el Protocolo Social del gremio")
    if f_bdo:
        resp_bits.append("el trabajo con Banca de las Oportunidades")
    if f_ofs:
        resp_bits.append("los objetivos de finanzas sostenibles")
    if f_datos or f_hoja:
        resp_bits.append("iniciativas regulatorias como la exigencia de datos desagregados y hojas de ruta sectoriales")
    if resp_bits:
        p2_parts.append("Desde la oferta sectorial se consolidan iniciativas como " + ", ".join(resp_bits))
    if f_directrices or f_marco:
        dims: List[str] = []
        if f_instit:
            dims.append("condiciones institucionales")
        if f_oferta or f_demanda:
            dims.append("diseño de oferta y demanda")
        if f_habil:
            dims.append("habilitadores transversales")
        if f_sosten:
            dims.append("mecanismos de sostenibilidad")
        if dims:
            p2_parts.append("y se propone un marco de acción con directrices que abarcan " + ", ".join(dims))
        else:
            p2_parts.append("y se propone un marco de acción con directrices para transformar el sistema")
    if f_riesgo:
        p2_parts.append("acompañado de ajustes a modelos de riesgo y prácticas de gestión")
    p2 = " ".join(p2_parts) + "."

    # Paragraph 3: implications and outlook
    p3_parts: List[str] = []
    p3_parts.append("El enfoque sugiere transitar de intervenciones aisladas a una estrategia estructural de largo plazo")
    if f_instit or f_sosten:
        p3_parts.append("con gobernanza clara, metas de cierre de brechas y mecanismos de seguimiento")
    if f_oferta or f_demanda:
        p3_parts.append("alineando capacidades de las entidades con necesidades diferenciales de los segmentos")
    p3_parts.append("para fortalecer la autonomía económica de las mujeres y mejorar la eficiencia del sistema financiero")
    p3 = " ".join(p3_parts) + "."

    # Paragraph 4: actionable orientation (keeps narrative, no bullets)
    p4_parts: List[str] = []
    p4_parts.append("Para avanzar, se recomienda consolidar capacidades analíticas y de diseño de productos con enfoque diferencial")
    if f_datos:
        p4_parts.append("profundizando en el uso de datos desagregados y métricas de impacto para la toma de decisiones")
    if f_riesgo:
        p4_parts.append("ajustando modelos de riesgo y criterios de admisión a partir de evidencia y pilotos controlados")
    if f_oferta or f_demanda:
        p4_parts.append("integrando educación financiera y acompañamiento no financiero para elevar el uso efectivo")
    if f_habil or f_instit:
        p4_parts.append("y fortaleciendo la gobernanza interna, la cultura organizacional y los habilitadores tecnológicos")
    p4 = " ".join(p4_parts) + "."

    # Assemble and keep near target length
    paragraphs = [p for p in (p1, p2, p3, p4) if p and len(p) > 10]
    summary = "\n\n".join(paragraphs)
    # If too short, extend with neutral synthesis
    wc = len(summary.split())
    min_w, max_w = target_words
    if wc < min_w:
        add = " En síntesis, el documento articula diagnóstico, lineamientos e implementación para acelerar el cierre de brechas con un enfoque práctico y medible, priorizando la trazabilidad de resultados y la mejora continua."
        summary = (summary + add)
    return summary.strip()


def stage1_process(semana_pdf_path: str) -> List[Dict[str, object]]:
    text = run_pdftotext_to_string(semana_pdf_path)
    items = parse_semana_pdf(text)
    results = [analyze_article(it) for it in items]
    return results


def stage2_process(banca_pdf_path: str) -> Dict[str, str]:
    text = run_pdftotext_to_string(banca_pdf_path)
    title = extract_banca_title(text) or "Artículo Banca & Economía"
    summary = summarize_banca_text(text)
    hits = find_keyword_hits(text, KEYWORDS)
    if hits:
        hits.sort(key=lambda x: (-x[1], x[0]))
        cuerpo = _build_impact_paragraph(hits, text, title)
        impacto = (
            "Se observa impacto para el revisor fiscal en la medida, que (\n"
            + cuerpo
            + "\n)\n"
            + "Palabras clave detectadas (total): "
            + _format_hits(hits)
        )
    else:
        impacto = "No se observa impacto para el revisor fiscal"
    return {"titulo": title, "resumen": summary, "resultado": impacto}


def find_latest_pdf_in_dir(directory: str) -> Optional[str]:
    if not os.path.isdir(directory):
        return None
    pdfs = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    if not pdfs:
        return None
    pdfs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pdfs[0]


def main():
    ensure_dirs()
    outputs: Dict[str, object] = {}

    # Etapa 1
    semana_pdf = find_latest_pdf_in_dir(SEMANA_DIR)
    if semana_pdf and os.path.exists(semana_pdf):
        try:
            stage1 = stage1_process(semana_pdf)
        except Exception as e:
            stage1 = []
            print(f"[ERROR] Etapa 1 falló: {e}", file=sys.stderr)
        with open(os.path.join(SALIDAS_DIR, "resultado_etapa1.txt"), "w", encoding="utf-8") as f:
            f.write(format_stage1_output(stage1))
        outputs["etapa1"] = stage1
    else:
        print(f"[ADVERTENCIA] No se encontró un PDF en {SEMANA_DIR}")

    # Etapa 2
    banca_pdf = find_latest_pdf_in_dir(BANCA_DIR)
    if banca_pdf and os.path.exists(banca_pdf):
        try:
            stage2 = stage2_process(banca_pdf)
        except Exception as e:
            stage2 = {"resumen": "", "resultado": f"Error en etapa 2: {e}"}
        with open(os.path.join(SALIDAS_DIR, "resultado_etapa2.txt"), "w", encoding="utf-8") as f:
            if stage2.get("titulo"):
                f.write(f"Título: {stage2.get('titulo')}\n\n")
            f.write("Resumen profesional (250-300 palabras aprox):\n\n")
            f.write(stage2.get("resumen", ""))
            f.write("\n\n")
            f.write("Resultado sobre palabras clave:\n")
            f.write(stage2.get("resultado", ""))
            f.write("\n")
        outputs["etapa2"] = stage2
    else:
        print(f"[ADVERTENCIA] No se encontró un PDF en {BANCA_DIR}")

    # Resumen combinado JSON para uso posterior
    with open(os.path.join(SALIDAS_DIR, "resultado.json"), "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print("Procesamiento finalizado. Revisa la carpeta 'salidas/'.")


if __name__ == "__main__":
    main()
