# PI Challenge — RAG con LLMs

API REST que responde preguntas sobre un documento usando **RAG** (*Retrieval Augmented
Generation*): recupera de una base vectorial el fragmento más relevante del documento y se
lo pasa como contexto a un LLM para que redacte la respuesta.

> **Estado:** en desarrollo. Ver [Estado del proyecto](#estado-del-proyecto).

---

## Índice

- [Cómo funciona](#cómo-funciona)
- [Stack](#stack)
- [Requisitos previos](#requisitos-previos)
- [Instalación paso a paso](#instalación-paso-a-paso)
- [Configuración](#configuración)
- [Ejecución](#ejecución)
- [Uso de la API](#uso-de-la-api)
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
| Misma pregunta → misma respuesta | `temperature = 0` (decodificación determinista) |
| Una sola oración | Instrucción explícita en el prompt |
| Mismo idioma que la pregunta | Instrucción en el prompt + embeddings multilingües |
| Emojis que resuman el contenido | Instrucción en el prompt |
| Siempre en tercera persona | Instrucción en el prompt |

---

## Stack

| Componente | Elección | Motivo |
|---|---|---|
| API | FastAPI | Validación con Pydantic, documentación automática, async nativo |
| LLM | Cohere (`command-*`) | Tier gratuito, buen soporte multilingüe |
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
| `COHERE_CHAT_MODEL` | `command-r-08-2024` | Modelo que redacta la respuesta |
| `DOCUMENT_PATH` | `data/documento.docx` | Documento que se indexa al arrancar |
| `TOP_K` | `1` | Cantidad de fragmentos a recuperar por consulta |
| `LLM_TEMPERATURE` | `0` | `0` = respuestas deterministas (requisito del challenge) |

---

## Ejecución

> ⏳ Pendiente — se completa cuando la aplicación esté implementada.

```bash
uvicorn app.api.main:app --reload
```

La API quedará disponible en `http://127.0.0.1:8000` y la documentación interactiva en
`http://127.0.0.1:8000/docs`.

---

## Uso de la API

> ⏳ Pendiente — se completa cuando los endpoints estén implementados.

**Request**

```json
{
  "user_name": "John Doe",
  "question": "¿Quién es Zara?"
}
```

**Preguntas de ejemplo**

- `¿Quién es Zara?` *(español)*
- `What did Emma decide to do?` *(inglés)*
- `What is the name of the magical flower?` *(inglés)*

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
├── requirements.txt      Dependencias con versión fijada.
└── README.md
```

La organización sigue **clean architecture**: las dependencias apuntan siempre hacia
adentro. El núcleo (`domain/`) define las interfaces que necesita, y las capas externas las
implementan. Así la lógica de negocio no conoce a Cohere ni a ChromaDB, lo que permite
cambiar de proveedor sin tocar el núcleo y testear sin llamadas de red.

---

## Tests

> ⏳ Pendiente.

```bash
pytest
```

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

---

## Estado del proyecto

- [x] Estructura del proyecto (clean architecture)
- [x] Configuración de entorno (`.env.example`, `.gitignore`)
- [x] Ambiente virtual y `requirements.txt`
- [x] Capa de dominio: modelos, puertos y chunking
- [x] Adaptadores: Cohere, ChromaDB, lectura de `.docx`
- [ ] Casos de uso y prompt
- [ ] API FastAPI
- [ ] Tests
- [ ] Dockerfile
- [ ] Colección de Postman
