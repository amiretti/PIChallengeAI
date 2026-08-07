# Imagen para levantar la API en un contenedor.
#
# Build en dos etapas: las dependencias se instalan en un ambiente virtual dentro de una
# etapa intermedia, y a la imagen final solo se copia ese ambiente ya construido. Así las
# herramientas de compilación y la caché de pip no llegan a la imagen que se distribuye.

# ---------------------------------------------------------------------------
# Etapa 1 — builder: instala las dependencias
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Opcional y vacío por defecto: un build normal no se ve afectado. Sirve para entornos
# donde un antivirus o un proxy corporativo intercepta HTTPS y re-firma los certificados
# (ver "Solución de problemas" en el README): sin esto, pip no confía en pypi.org y el
# build falla. Se pasa con:  --build-arg EXTRA_CA_CERT="$(cat mi-ca.crt)"
ARG EXTRA_CA_CERT=""
RUN if [ -n "$EXTRA_CA_CERT" ]; then \
        printf '%s\n' "$EXTRA_CA_CERT" > /usr/local/share/ca-certificates/extra-ca.crt && \
        update-ca-certificates && \
        pip config set global.cert /etc/ssl/certs/ca-certificates.crt; \
    fi

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Se copia solo requirements.txt: mientras no cambie, Docker reutiliza esta capa y se
# saltea la instalación completa, que es la parte lenta del build.
COPY requirements.txt .
RUN pip install --requirement requirements.txt

# ---------------------------------------------------------------------------
# Etapa 2 — runtime: la imagen que efectivamente se ejecuta
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Usuario sin privilegios: si el proceso se ve comprometido, no es root del contenedor.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser app/ app/
COPY --chown=appuser:appuser data/ data/

USER appuser

EXPOSE 8000

# El endpoint /health existe para esto. `start-period` es generoso porque el arranque
# indexa el documento contra la API de embeddings antes de aceptar tráfico.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request as r; r.urlopen('http://127.0.0.1:8000/health').read()"

# --host 0.0.0.0 es obligatorio: con el default (127.0.0.1) el servidor solo escucharía
# dentro del contenedor y el puerto publicado no respondería desde afuera.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
