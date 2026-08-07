# PI Challenge — RAG con LLMs

API REST que responde preguntas sobre un documento usando **RAG** (*Retrieval Augmented
Generation*): recupera de una base vectorial el fragmento más relevante del documento y se
lo pasa como contexto a un LLM para que redacte la respuesta.

> **Estado:** completo, incluidos los puntos opcionales (Dockerfile y colección de
> Postman). Ver [Estado del proyecto](#estado-del-proyecto).

---

## Índice

- [Cómo funciona](#cómo-funciona)
- [Stack](#stack)
- [Requisitos previos](#requisitos-previos)
- [Instalación paso a paso](#instalación-paso-a-paso)
- [Configuración](#configuración)
- [Ejecución](#ejecución)
- [Ejecución con Docker](#ejecución-con-docker)
- [Uso de la API](#uso-de-la-api)
- [Colección de Postman](#colección-de-postman)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Tests](#tests)
- [Estado del proyecto](#estado-del-proyecto)

---

## Cómo funciona

```
INDEXADO (una vez, al iniciar la aplicación)

  documento.docx
      │
      ├─► Parseo        extraer el texto plano
      ├─► Chunking      dividir en 5 fragmentos (uno por párrafo)
      ├─► Embedding     convertir cada fragmento en un vector
      └─► ChromaDB      almacenar {id, texto, vector}


CONSULTA (en cada request HTTP)

  POST /  {"user_name": "John Doe", "question": "¿Quién es Zara?"}
      │
      ├─► Caché         si la pregunta ya se respondió, devolver esa misma respuesta
      ├─► Embedding     convertir la pregunta en un vector
      ├─► Búsqueda      recuperar el fragmento más similar
      ├─► Prompt        reglas + contexto recuperado + pregunta
      ├─► LLM           generar la respuesta
      │
      └─► {"answer": "Zara es una intrépida exploradora... 🚀🌌"}
```

El LLM no busca ni recuerda: solo redacta usando el fragmento que el sistema le entrega.
Eso evita que invente información que no está en el documento.

### Requisitos de la respuesta

| Requisito | Cómo se garantiza |
|---|---|
| Misma pregunta → misma respuesta | Caché de respuestas por pregunta (ver [Determinismo](#determinismo)) + `temperature = 0` y `seed` fijo |
| Una sola oración | Instrucción explícita en el prompt |
| Mismo idioma que la pregunta | Instrucción reforzada en el prompt + modelo `command-a` (ver [Idioma](#idioma-de-la-respuesta)) |
| Emojis que resuman el contenido | Instrucción en el prompt |
| Siempre en tercera persona | Instrucción en el prompt |

Los cinco requisitos están verificados por tests de integración contra la API real
(`tests/test_rag_pipeline_integration.py`): son los únicos que pueden probar el
comportamiento de un LLM.

### Determinismo

`temperature = 0` y un `seed` fijo **no alcanzan**. Las APIs de LLM alojadas no garantizan
salida reproducible bit a bit: el batching en GPU hace que la misma entrada pueda producir
salidas levemente distintas. Se verificó experimentalmente con los tres modelos de Cohere
evaluados — la oración se mantenía, pero los emojis cambiaban entre llamadas idénticas:

```
Q: ¿Quién es Zara?
A1: Zara es un explorador intrépido que busca la paz en la galaxia. 🌌🤝🚀
A2: Zara es un explorador intrépido que busca la paz en la galaxia. 🌟🌌🤝
```

Por eso el determinismo se resuelve en la aplicación, no en el modelo: `AnswerQuestion`
cachea la respuesta por pregunta normalizada (sin distinguir mayúsculas ni espaciado). La
primera vez se consulta al LLM; a partir de ahí, la misma pregunta devuelve exactamente el
mismo texto. Es una garantía del sistema, no una expectativa sobre el modelo — y además
evita llamadas repetidas a la API.

> El nombre del usuario queda deliberadamente fuera del prompt y de la clave de caché: si
> influyera, la misma pregunta hecha por dos personas podría responderse distinto.

**Limitaciones conocidas**, explícitas para no dar una garantía más fuerte de la real:

- La caché vive en memoria del proceso y **no tiene límite de tamaño**. Para este alcance
  (un documento, uso local) es lo correcto: acotarla con desalojo rompería el determinismo
  justo para las preguntas desalojadas. En un servicio expuesto habría que revisarlo, ya
  que la clave proviene de input del usuario.
- La caché y el índice son **por proceso**. Con un solo worker —el modo por defecto— la
  garantía se cumple. Levantar `uvicorn --workers N` la rompe: cada worker mantiene su
  propia caché, y la misma pregunta atendida por dos workers distintos podría devolver
  textos distintos. Escalar horizontalmente exigiría mover ambos a un almacén compartido.

### Idioma de la respuesta

El documento está en español, así que el contexto recuperado también lo está. Los modelos
tienden a responder en el idioma del **contexto** en lugar del de la **pregunta**. Se probaron
tres redacciones distintas del prompt y las tres fallaron en inglés con `command-r`:

| Modelo | `Who is Zara?` | `Quem são os Dracorians?` |
|---|---|---|
| `command-r-08-2024` | ❌ responde en español | ✅ portugués |
| `command-r-plus-08-2024` | ❌ responde en español | ✅ portugués |
| `command-a-03-2025` | ✅ inglés | ✅ portugués |

Por eso el modelo por defecto es `command-a-03-2025`. El prompt además repite la regla de
idioma **después** de la pregunta, donde tiene más peso que en las instrucciones iniciales.

---

## Stack

| Componente | Elección | Motivo |
|---|---|---|
| API | FastAPI | Validación con Pydantic, documentación automática, async nativo |
| LLM | Cohere `command-a-03-2025` | Tier gratuito; es el único de los evaluados que respeta el idioma de la pregunta |
| Embeddings | Cohere `embed-multilingual-v3.0` | El documento está en español y las preguntas pueden llegar en inglés o portugués |
| Base vectorial | ChromaDB | Embebida, sin servidor, integración nativa con Cohere |
| Orquestación | Código propio | Sin LangChain — el pipeline es lineal y las abstracciones propias mantienen limpia la arquitectura |

---

## Requisitos previos

1. **Python 3.12 o superior**

   Verificar la versión instalada:

   ```bash
   python --version
   ```

   Si el comando no existe o la versión es menor, descargarlo desde
   [python.org/downloads](https://www.python.org/downloads/).
   En Windows, marcar **"Add Python to PATH"** durante la instalación.

2. **Git**

   ```bash
   git --version
   ```

   Si no está instalado: [git-scm.com/downloads](https://git-scm.com/downloads).

3. **Una API key de Cohere** (gratuita)

   Crear una cuenta en [cohere.com](https://cohere.com) y generar una clave en
   [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys).
   El *trial key* gratuito alcanza para probar el proyecto.

---

## Instalación paso a paso

### 1. Clonar el repositorio

```bash
git clone https://github.com/amiretti/PIChallengeAI.git
cd PIChallengeAI
```

### 2. Crear un ambiente virtual

Un **ambiente virtual** es una carpeta aislada con su propia copia de Python y sus propios
paquetes. Evita que las dependencias de este proyecto se mezclen con las de otros proyectos
o con la instalación global del sistema.

```bash
python -m venv .venv
```

Esto crea la carpeta `.venv/`, que está excluida del repositorio (no se versiona).

### 3. Activar el ambiente virtual

**Windows — PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

> Si aparece el error `no se puede cargar el archivo ... porque la ejecución de scripts
> está deshabilitada`, habilitarlos para el usuario actual con:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

**Windows — CMD:**

```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

Al activarse, el prompt de la terminal muestra el prefijo `(.venv)`. Ésa es la señal de
que todo lo que se instale a partir de ahora va dentro del entorno aislado.

Para salir del entorno en cualquier momento: `deactivate`.

### 4. Instalar las dependencias

```bash
pip install -r requirements.txt
```

`requirements.txt` lista las librerías del proyecto con su versión exacta. Fijar las
versiones garantiza que todos instalen exactamente lo mismo que fue probado, y que una
actualización futura de alguna librería no rompa el proyecto.

La instalación descarga alrededor de 200 MB (ChromaDB incluye dependencias de cómputo
numérico) y puede demorar unos minutos la primera vez.

---

## Configuración

El proyecto lee su configuración de variables de entorno. Nunca hay credenciales en el
código fuente.

### 1. Crear el archivo `.env`

**Windows — PowerShell:**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux:**

```bash
cp .env.example .env
```

### 2. Completar la API key

Abrir `.env` y pegar la clave de Cohere:

```dotenv
COHERE_API_KEY=tu-api-key-aca
```

> ⚠️ El archivo `.env` está en `.gitignore` y **nunca** debe subirse al repositorio.
> `.env.example` sí se versiona: documenta qué variables hacen falta, sin exponer valores.

### Variables disponibles

| Variable | Default | Descripción |
|---|---|---|
| `COHERE_API_KEY` | *(requerida)* | Clave de la API de Cohere |
| `COHERE_EMBEDDING_MODEL` | `embed-multilingual-v3.0` | Modelo de embeddings. Debe ser multilingüe |
| `COHERE_CHAT_MODEL` | `command-a-03-2025` | Modelo que redacta la respuesta. Ver [Idioma](#idioma-de-la-respuesta) antes de cambiarlo |
| `DOCUMENT_PATH` | `data/documento.docx` | Documento que se indexa al arrancar |
| `TOP_K` | `1` | Cantidad de fragmentos a recuperar por consulta |
| `LLM_TEMPERATURE` | `0` | `0` = decodificación determinista (necesaria, pero no suficiente) |
| `LLM_SEED` | `42` | Semilla del modelo. Reduce la variación, sin eliminarla |

---

## Ejecución

Con el ambiente virtual activado y el `.env` completo:

```bash
uvicorn app.api.main:app --reload
```

La API queda disponible en `http://127.0.0.1:8000` y la documentación interactiva en
`http://127.0.0.1:8000/docs`.

Al arrancar, la aplicación lee el documento, lo divide en chunks y los indexa en ChromaDB.
Eso ocurre **una sola vez**, antes de aceptar requests: si el documento no existe o la API
key es inválida, el servidor falla al iniciar en vez de devolver errores request a request.
El log lo confirma:

```
INFO:     Waiting for application startup.
INFO:     Indexed 5 chunks from data/documento.docx
INFO:     Application startup complete.
```

---

## Ejecución con Docker

Alternativa a la instalación local: no requiere tener Python 3.12 ni instalar dependencias.
Sí requiere Docker y una API key de Cohere.

### 1. Construir la imagen

```bash
docker build -t pichallenge-rag .
```

### 2. Levantar el contenedor

```bash
docker run --rm -p 8000:8000 --env-file .env pichallenge-rag
```

La API queda en `http://127.0.0.1:8000`, igual que en la ejecución local.

Si no se quiere usar un archivo `.env`, la clave puede pasarse directamente:

```bash
docker run --rm -p 8000:8000 -e COHERE_API_KEY=tu-api-key-aca pichallenge-rag
```

> 🔑 La API key se pasa **en tiempo de ejecución**, nunca se hornea en la imagen. El
> archivo `.env` está excluido en `.dockerignore`, así que no llega al contenedor aunque
> exista en el directorio del proyecto.

### Detalles de la imagen

| Decisión | Motivo |
|---|---|
| Build en dos etapas | Las dependencias se compilan en una etapa intermedia; a la imagen final solo se copia el ambiente virtual ya armado, sin caché de pip ni herramientas de build |
| Usuario `appuser` (uid 1000) | El proceso no corre como root: si se ve comprometido, no es administrador del contenedor |
| `HEALTHCHECK` sobre `/health` | Docker reporta el contenedor como `healthy` recién cuando la API responde. El `start-period` es amplio porque el arranque indexa el documento contra la API de embeddings |
| `--host 0.0.0.0` | Con el default (`127.0.0.1`) uvicorn solo escucharía dentro del contenedor y el puerto publicado no respondería |
| `COPY requirements.txt` primero | Mientras las dependencias no cambien, Docker reutiliza la capa de instalación, que es la parte lenta del build |

La imagen pesa unos 820 MB, dominados por ChromaDB y sus dependencias de cómputo numérico
(ONNX Runtime).

---

## Uso de la API

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/` | Responde una pregunta sobre el documento |
| `GET` | `/health` | Verifica que el servicio está levantado |

### `POST /`

**Request**

```json
{
  "user_name": "John Doe",
  "question": "¿Quién es Zara?"
}
```

Ambos campos son obligatorios y no pueden estar vacíos. Los espacios sobrantes se recortan.

**Response — `200 OK`**

```json
{
  "answer": "Zara es un intrépido explorador que descubre un antiguo artefacto que podría contener la clave para la paz entre los Dracorians y los Lumis. 🌌🚀🔍"
}
```

**Ejemplo con `curl`**

```bash
curl -X POST http://127.0.0.1:8000/ \
  -H "Content-Type: application/json" \
  -d '{"user_name": "John Doe", "question": "What did Emma decide to do?"}'
```

### Preguntas de ejemplo

Respuestas reales del servicio:

| Pregunta | Respuesta |
|---|---|
| `¿Quién es Zara?` | Zara es un intrépido explorador que descubre un antiguo artefacto que podría contener la clave para la paz entre los Dracorians y los Lumis. 🌌🚀🔍 |
| `What did Emma decide to do?` | Emma decided to share her extra day with the village, leaving an indelible mark on the heart of every inhabitant. 🌟🎁💖 |
| `What is the name of the magical flower?` | The magical flower is called "Luz de Luna." 🌙🌿🌟 |
| `Quem são os Dracorians?` | Os Dracorians são uma das duas civilizações alienígenas da galáxia de Zenthoria, que estão à beira de uma guerra intergaláctica com os Lumis. 🌌⚔️ |

Ante una pregunta que el documento no cubre, no inventa:

| Pregunta | Respuesta |
|---|---|
| `¿Quién ganó el mundial 2022?` | El documento no menciona quién ganó el mundial 2022. 🤷‍♂️🌍⚽ |

### Códigos de error

| Código | Cuándo |
|---|---|
| `422` | El request no cumple el contrato: falta un campo, está vacío o tiene el tipo incorrecto |
| `502` | Falló una dependencia externa (Cohere o ChromaDB) |

Las respuestas `502` devuelven un mensaje genérico. El detalle del error se registra en el
log del servidor y nunca se envía al cliente: los errores de Cohere incluyen cabeceras de
la petición y datos de la cuenta.

---

## Colección de Postman

En `postman/` hay una colección lista para importar, con la API corriendo en
`http://127.0.0.1:8000`:

| Archivo | Contenido |
|---|---|
| `PI-Challenge-RAG-API.postman_collection.json` | Los 10 requests con sus tests |
| `PI-Challenge-local.postman_environment.json` | Variable `baseUrl` apuntando al entorno local |

**Importar:** en Postman, *Import* → seleccionar ambos archivos → elegir el environment
«PI Challenge — local» en el selector de arriba a la derecha.

No es solo una lista de requests: cada uno lleva tests que verifican los requisitos del
enunciado, no únicamente el código de estado.

| Carpeta | Qué comprueba |
|---|---|
| **Health check** | El servicio está levantado |
| **Preguntas del challenge** | Las tres preguntas de ejemplo más una en portugués. Cada una valida que la respuesta sea una sola oración, con emojis, en tercera persona y **en el idioma de la pregunta** |
| **Requisitos de la respuesta** | Determinismo (repite la pregunta con otro usuario y compara texto exacto) y que no invente ante una pregunta fuera del documento |
| **Validación del request** | Los tres casos que deben rechazarse con `422` |

> Conviene ejecutar la colección completa (*Run collection*) en vez de requests sueltos: el
> test de determinismo compara contra la respuesta que guardó el request «¿Quién es Zara?».

### Ejecución por línea de comandos

La colección también corre sin la interfaz de Postman, con
[newman](https://github.com/postmanlabs/newman):

```bash
npx newman run postman/PI-Challenge-RAG-API.postman_collection.json \
  -e postman/PI-Challenge-local.postman_environment.json \
  --timeout-request 90000
```

Salida de una corrida real contra el servicio:

```
requests      10    0 failed
assertions    36    0 failed
```

> El `--timeout-request` amplio es necesario porque una pregunta nueva implica dos llamadas
> a Cohere (embedding + generación) y puede tardar más que el default de Postman. Una
> pregunta ya respondida vuelve de la caché en milisegundos.

---

## Estructura del proyecto

```
PIChallengeAI/
├── app/
│   ├── domain/           Núcleo: modelos, interfaces (puertos) y lógica de chunking.
│   │                     No depende de ninguna librería externa.
│   ├── application/      Casos de uso y plantilla del prompt.
│   ├── infrastructure/   Adaptadores concretos: Cohere, ChromaDB, lectura de .docx.
│   └── api/              Capa HTTP: endpoints, schemas y wiring de dependencias.
├── data/
│   └── documento.docx    Documento fuente que indexa la aplicación.
├── docs/
│   └── instructions/     Material de referencia (enunciado del challenge, notebook).
├── tests/
├── postman/              Colección para probar la API.
├── .env.example          Plantilla de configuración.
├── .dockerignore         Excluye el .env y el entorno local de la imagen.
├── Dockerfile            Build en dos etapas para levantar la API en un contenedor.
├── requirements.txt      Dependencias con versión fijada.
└── README.md
```

La organización sigue **clean architecture**: las dependencias apuntan siempre hacia
adentro. El núcleo (`domain/`) define las interfaces que necesita, y las capas externas las
implementan. Así la lógica de negocio no conoce a Cohere ni a ChromaDB, lo que permite
cambiar de proveedor sin tocar el núcleo y testear sin llamadas de red.

---

## Tests

```bash
pip install -r requirements-dev.txt
```

Los tests están separados en dos grupos:

```bash
pytest -m "not integration"   # unitarios: rápidos, sin red ni API key
pytest -m integration         # integración: requieren COHERE_API_KEY y red
pytest                        # todos
```

Los **unitarios** cubren chunking, prompt, casos de uso y endpoints. Usan dobles de prueba
(`tests/fakes.py`) que implementan los puertos del dominio, y los endpoints se prueban con
el `TestClient` de FastAPI sobreescribiendo la dependencia del caso de uso. No tocan la red.

Los de **integración** son los que verifican los requisitos de la respuesta contra la API
real de Cohere: sin ellos no hay forma de comprobar que el LLM responde en una oración, en
el idioma correcto, en tercera persona y con emojis.

> ⚠️ El *trial key* de Cohere admite 20 llamadas por minuto. Correr los dos archivos de
> integración juntos puede superar ese límite; en ese caso, ejecutarlos por separado.

---

## Solución de problemas

### `CERTIFICATE_VERIFY_FAILED` al llamar a Cohere

Algunos antivirus (Avast, Kaspersky, ESET) y proxies corporativos interceptan el tráfico
HTTPS y lo re-firman con una CA propia. Esa CA se instala en el almacén de certificados del
sistema operativo — por eso los navegadores funcionan — pero Python usa su propio bundle
(`certifi`) y rechaza la conexión.

Para que Python confíe también en el almacén del sistema:

```bash
pip install pip-system-certs
```

No está en `requirements.txt` porque es una particularidad del entorno local, no una
dependencia del proyecto. La alternativa es desactivar el escaneo HTTPS del antivirus.

### `CERTIFICATE_VERIFY_FAILED` usando Docker

**Docker no evita este problema.** El contenedor tiene su propio almacén de certificados,
que tampoco contiene la CA del antivirus o del proxy. El problema aparece en dos momentos
distintos y cada uno se resuelve por separado.

Primero hay que exportar la CA que está interceptando. Para identificarla:

```bash
echo | openssl s_client -connect pypi.org:443 -servername pypi.org 2>/dev/null \
  | openssl x509 -noout -issuer
```

En Windows, exportarla desde el almacén del sistema:

```powershell
$cert = Get-ChildItem Cert:\LocalMachine\Root |
        Where-Object { $_.Subject -like "*Avast*" } | Select-Object -First 1
$b64 = [Convert]::ToBase64String($cert.RawData, 'InsertLineBreaks')
"-----BEGIN CERTIFICATE-----`n$b64`n-----END CERTIFICATE-----" |
        Set-Content -Path ca.crt -Encoding ascii
```

**1. Durante el build**, al bajar las dependencias de PyPI. El `Dockerfile` acepta un
argumento opcional para inyectar la CA:

```bash
docker build -t pichallenge-rag --build-arg EXTRA_CA_CERT="$(cat ca.crt)" .
```

**2. Durante la ejecución**, al llamar a la API de Cohere. Se resuelve montando un bundle
que combine los certificados estándar con la CA interceptora, y apuntando `SSL_CERT_FILE`
a ese archivo:

```bash
cat "$(python -c 'import certifi; print(certifi.where())')" ca.crt > ca-bundle.pem

docker run --rm -p 8000:8000 --env-file .env \
  -v "$(pwd)/ca-bundle.pem:/certs/ca-bundle.pem:ro" \
  -e SSL_CERT_FILE=/certs/ca-bundle.pem \
  pichallenge-rag
```

En un entorno sin interceptación TLS nada de esto hace falta: el `build` y el `run` de la
sección [Ejecución con Docker](#ejecución-con-docker) funcionan tal cual. La alternativa,
como en el caso local, es desactivar el escaneo HTTPS del antivirus.

---

## Estado del proyecto

- [x] Estructura del proyecto (clean architecture)
- [x] Configuración de entorno (`.env.example`, `.gitignore`)
- [x] Ambiente virtual y `requirements.txt`
- [x] Capa de dominio: modelos, puertos y chunking
- [x] Adaptadores: Cohere, ChromaDB, lectura de `.docx`
- [x] Capa de aplicación: casos de uso y prompt
- [x] Tests de dominio y aplicación (unitarios + integración)
- [x] API FastAPI
- [x] Tests de la API
- [x] Dockerfile
- [x] Colección de Postman
