import os
import io
import time
from datetime import datetime

import streamlit as st

# Reuse your processing logic from main.py
from main import (
    ensure_dirs,
    stage1_process,
    stage2_process,
    format_stage1_output,
    find_latest_pdf_in_dir,
    SEMANA_DIR,
    BANCA_DIR,
    SALIDAS_DIR,
    CACHE_DIR,
    ANEXOS_DIR,
    shutil,
)


def _save_uploaded(file, target_dir: str, prefix: str) -> str:
    os.makedirs(target_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = os.path.basename(file.name)
    out_path = os.path.join(target_dir, f"{prefix}_{ts}_{safe_name}")
    with open(out_path, "wb") as f:
        f.write(file.getbuffer())
    return out_path


def _txt_bytes(s: str) -> bytes:
    return s.encode("utf-8")


def _check_pdftotext_available() -> bool:
    return bool(shutil.which("pdftotext"))


def ui_sidebar():
    st.sidebar.header("Estado del entorno")
    st.sidebar.write(f"Directorio de trabajo: `{os.path.abspath(os.getcwd())}`")
    st.sidebar.write(f"Carpeta 'Informe Asobancaria': `{SEMANA_DIR}`")
    st.sidebar.write(f"Carpeta 'Banca y Economia': `{BANCA_DIR}`")
    st.sidebar.write(f"Carpeta de salidas: `{SALIDAS_DIR}`")
    st.sidebar.write(f"Caché de anexos: `{ANEXOS_DIR}`")

    ok_pdf = _check_pdftotext_available()
    st.sidebar.markdown(
        ("✅ `pdftotext` disponible" if ok_pdf else "⚠️ `pdftotext` NO disponible. Instala `poppler-utils`.")
    )
    if not ok_pdf:
        st.sidebar.info(
            "Linux/Debian/Ubuntu: `sudo apt-get update && sudo apt-get install -y poppler-utils`\n"
            "macOS (brew): `brew install poppler`"
        )

    st.sidebar.divider()
    if st.sidebar.button("Vaciar caché de anexos"):
        try:
            if os.path.isdir(ANEXOS_DIR):
                for name in os.listdir(ANEXOS_DIR):
                    p = os.path.join(ANEXOS_DIR, name)
                    if os.path.isfile(p):
                        os.remove(p)
            st.sidebar.success("Caché vaciada")
        except Exception as e:
            st.sidebar.error(f"No se pudo limpiar la caché: {e}")


def page_etapa1():
    st.subheader("Etapa 1 – Informe Semanal Asobancaria")
    st.write("Sube el PDF del Informe Semanal o usa el más reciente en la carpeta.")

    uploaded = st.file_uploader("PDF Informe Semanal", type=["pdf"], key="semana_pdf")
    col1, col2 = st.columns(2)
    with col1:
        use_latest = st.checkbox("Usar PDF más reciente del directorio", value=(uploaded is None))
    with col2:
        process_btn = st.button("Procesar etapa 1")

    pdf_path = None
    if uploaded and not use_latest:
        pdf_path = _save_uploaded(uploaded, SEMANA_DIR, prefix="semana")
        st.caption(f"Guardado en: {pdf_path}")
    elif use_latest:
        pdf_path = find_latest_pdf_in_dir(SEMANA_DIR)
        if pdf_path:
            st.caption(f"Usando más reciente: {pdf_path}")

    if process_btn:
        ensure_dirs()
        if not pdf_path:
            st.error("No hay PDF para procesar.")
            return
        if not _check_pdftotext_available():
            st.error("`pdftotext` no está disponible. Instala `poppler-utils` y vuelve a intentar.")
            return
        with st.spinner("Procesando…"):
            try:
                results = stage1_process(pdf_path)
            except Exception as e:
                st.error(f"Error en etapa 1: {e}")
                return

        if not results:
            st.warning("No se detectaron artículos o no hubo resultados.")
            return

        # Mostrar resultados
        st.success(f"{len(results)} artículo(s) procesado(s)")
        for idx, item in enumerate(results, 1):
            with st.expander(f"Artículo {idx}: {item.get('titulo') or '(sin título)'}"):
                st.write({
                    "Sección": item.get("seccion"),
                    "Título": item.get("titulo"),
                    "Resumen": item.get("resumen"),
                    "Fecha": item.get("fecha"),
                    "Enlaces": item.get("enlaces"),
                })
                st.markdown("**Resultado:**")
                st.write(item.get("resultado") or "")

        # Descargas
        txt = format_stage1_output(results)
        st.download_button(
            label="Descargar TXT etapa 1",
            data=_txt_bytes(txt),
            file_name="resultado_etapa1.txt",
            mime="text/plain",
        )


def page_etapa2():
    st.subheader("Etapa 2 – Banca & Economía")
    st.write("Sube el PDF de Banca & Economía o usa el más reciente en la carpeta.")

    uploaded = st.file_uploader("PDF Banca & Economía", type=["pdf"], key="banca_pdf")
    col1, col2 = st.columns(2)
    with col1:
        use_latest = st.checkbox("Usar PDF más reciente del directorio", value=(uploaded is None), key="banca_latest")
    with col2:
        process_btn = st.button("Procesar etapa 2")

    pdf_path = None
    if uploaded and not use_latest:
        pdf_path = _save_uploaded(uploaded, BANCA_DIR, prefix="banca")
        st.caption(f"Guardado en: {pdf_path}")
    elif use_latest:
        pdf_path = find_latest_pdf_in_dir(BANCA_DIR)
        if pdf_path:
            st.caption(f"Usando más reciente: {pdf_path}")

    if process_btn:
        ensure_dirs()
        if not pdf_path:
            st.error("No hay PDF para procesar.")
            return
        if not _check_pdftotext_available():
            st.error("`pdftotext` no está disponible. Instala `poppler-utils` y vuelve a intentar.")
            return
        with st.spinner("Procesando…"):
            try:
                result = stage2_process(pdf_path)
            except Exception as e:
                st.error(f"Error en etapa 2: {e}")
                return

        st.success("Procesamiento completado")
        if result.get("titulo"):
            st.markdown(f"**Título**: {result['titulo']}")
        st.markdown("**Resumen profesional:**")
        st.write(result.get("resumen") or "")
        st.markdown("**Resultado sobre palabras clave:**")
        st.write(result.get("resultado") or "")

        # Descargas
        txt_buf = io.StringIO()
        if result.get("titulo"):
            txt_buf.write(f"Título: {result['titulo']}\n\n")
        txt_buf.write("Resumen profesional (250-300 palabras aprox):\n\n")
        txt_buf.write(result.get("resumen", ""))
        txt_buf.write("\n\n")
        txt_buf.write("Resultado sobre palabras clave:\n")
        txt_buf.write(result.get("resultado", ""))
        txt_buf.write("\n")
        st.download_button(
            label="Descargar TXT etapa 2",
            data=_txt_bytes(txt_buf.getvalue()),
            file_name="resultado_etapa2.txt",
            mime="text/plain",
        )


def main():
    st.set_page_config(page_title="Extractor Asobancaria", layout="wide")
    ensure_dirs()
    ui_sidebar()

    st.title("Extractor Asobancaria – Interfaz Web")
    st.caption("Cargue PDFs y ejecute las etapas sin tocar el código.")

    tabs = st.tabs(["Etapa 1", "Etapa 2"])  # simple por ahora
    with tabs[0]:
        page_etapa1()
    with tabs[1]:
        page_etapa2()


if __name__ == "__main__":
    main()

