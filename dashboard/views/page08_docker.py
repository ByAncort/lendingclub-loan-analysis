import streamlit as st
import os
import platform
import subprocess
import datetime
import time
import pandas as pd
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
from dashboard.utils.charts import kpi_card


def _run_cmd(cmd):
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or out.stderr.strip()
    except Exception:
        return None


def _inside_docker():
    return os.path.exists("/.dockerenv") or "docker" in (os.environ.get("container", "") or "").lower()


def show():
    st.title("Infraestructura Docker")
    st.markdown("Estado del contenedor, recursos del sistema y verificacion del entorno de despliegue.")

    in_docker = _inside_docker()

    st.subheader("Estado del Contenedor")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        status_color = "green" if in_docker else "orange"
        st.metric("Entorno Docker", "\U0001f4e6 DENTRO" if in_docker else "\U0001f4bb FUERA")
        st.caption("Contenedor Docker detectado" if in_docker else "Ejecucion directa")
    with c2:
        hostname = platform.node()
        st.metric("Hostname", hostname)
    with c3:
        py_ver = platform.python_version()
        st.metric("Python", py_ver)
    with c4:
        os_name = f"{platform.system()} {platform.release()}"
        st.metric("SO", os_name)

    st.markdown("---")

    st.subheader("Recursos del Sistema")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if HAS_PSUTIL:
            cpu = psutil.cpu_percent(interval=0.3)
            st.plotly_chart(
                kpi_card(cpu, "CPU %", fmt=".1f", suffix="%"),
                use_container_width=True,
            )
        else:
            st.metric("CPU %", "N/D")
            st.caption("psutil no instalado")
    with c2:
        if HAS_PSUTIL:
            mem = psutil.virtual_memory()
            mem_used = mem.used / (1024**3)
            mem_total = mem.total / (1024**3)
            st.plotly_chart(
                kpi_card(mem_used, "RAM Usada (GB)", fmt=".1f", suffix=f" / {mem_total:.1f}"),
                use_container_width=True,
            )
        else:
            st.metric("RAM Usada", "N/D")
    with c3:
        if HAS_PSUTIL:
            disk = psutil.disk_usage("/")
            disk_used = disk.used / (1024**3)
            disk_total = disk.total / (1024**3)
            st.plotly_chart(
                kpi_card(disk_used, "Disco Usado (GB)", fmt=".1f", suffix=f" / {disk_total:.1f}"),
                use_container_width=True,
            )
        else:
            st.metric("Disco Usado", "N/D")
    with c4:
        if HAS_PSUTIL:
            uptime_sec = time.time() - psutil.boot_time()
            uptime_hrs = uptime_sec / 3600
            st.plotly_chart(kpi_card(uptime_hrs, "Uptime Sistema", fmt=".1f", suffix=" hrs"), use_container_width=True)
        else:
            st.metric("Uptime", "N/D")

    st.markdown("---")

    st.subheader("Verificacion de Archivos Clave")
    required_paths = {
        "Dockerfile": "Dockerfile",
        "docker-compose.yml": "docker-compose.yml",
        ".dockerignore": ".dockerignore",
        "requirements.txt": "requirements.txt",
        "Modelo XGBoost": os.path.join("models", "xgboost_pipeline.pkl"),
        "Datos Aceptados": os.path.join("data", "processed", "accepted_clean.csv"),
        "Datos Rechazados": os.path.join("data", "processed", "rejected_clean.csv"),
        "Dashboard Entry": os.path.join("dashboard", "app.py"),
    }

    base = os.getcwd()
    rows = []
    for label, rel_path in required_paths.items():
        full = os.path.join(base, rel_path)
        exists = os.path.exists(full)
        size = ""
        if exists and os.path.isfile(full):
            sz = os.path.getsize(full)
            if sz >= 1024**3:
                size = f"{sz / 1024**3:.2f} GB"
            elif sz >= 1024**2:
                size = f"{sz / 1024**2:.2f} MB"
            elif sz >= 1024:
                size = f"{sz / 1024:.2f} KB"
            else:
                size = f"{sz} B"
        rows.append({
            "Archivo": label,
            "Ruta": rel_path,
            "Existe": "SI" if exists else "NO",
            "Tamano": size if exists else "-",
        })

    import pandas as pd
    df = pd.DataFrame(rows)

    def color_exists(val):
        color = "#2ecc71" if val == "SI" else "#e74c3c"
        return f"background-color: {color}; color: white"

    st.dataframe(
        df.style.map(color_exists, subset=["Existe"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    st.subheader("Variables de Entorno Relevantes")
    env_vars = {
        k: v for k, v in sorted(os.environ.items())
        if any(prefix in k.lower() for prefix in ("docker", "container", "kube", "host", "path", "python", "streamlit"))
    }
    if env_vars:
        env_df = pd.DataFrame({"Variable": list(env_vars.keys()), "Valor": list(env_vars.values())})
        st.dataframe(env_df, use_container_width=True, hide_index=True)
    else:
        st.info("No se detectaron variables de entorno relacionadas con Docker/entorno.")

    st.markdown("---")

    st.subheader("Comandos Docker")
    st.markdown("""
| Comando | Descripcion |
|---------|-------------|
| `docker build -t lendingclub-dashboard .` | Construir la imagen |
| `docker-compose up -d` | Iniciar el contenedor en segundo plano |
| `docker-compose down` | Detener y eliminar el contenedor |
| `docker-compose logs -f` | Ver logs en tiempo real |
| `docker ps` | Listar contenedores activos |
| `docker stats` | Monitorear uso de recursos |
    """)

    st.markdown("---")
    st.caption(
        "Entorno: **Docker**" if in_docker else "Entorno: **Local** - Ejecute `docker-compose up -d` para desplegar en contenedor."
    )
