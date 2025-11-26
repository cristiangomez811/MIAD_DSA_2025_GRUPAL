# Manual de Usuario – Instalación del Sistema

Este manual describe los pasos necesarios para instalar y poner en funcionamiento el sistema desarrollado para el proyecto de predicción de satisfacción de clientes. El objetivo es que el usuario pueda preparar correctamente el entorno y ejecutar tanto el modelo entrenado como el tablero interactivo de visualización.

## 1.	 Requisitos previos
Antes de iniciar la instalación del sistema, asegúrese de contar con los siguientes elementos:
Sistema operativo compatible: Windows 10 o superior, macOS 11 o superior, Linux (Ubuntu recomendado)
1.1.	Python 3.12 o superior: Lo cual puede verificarse ejecutando en la terminal:
```
python --version
```

### 1.2.	Git (para clonar el repositorio): Lo cual puede verificarse ejecutando en la terminal:
```
git --version
```
### 1.3.	Conexión a internet para instalar dependencias del proyecto.

## 2.	 Descarga del proyecto
Clone el repositorio ejecutando el siguiente comando en la terminal:
```
git clone <URL-del-repositorio>
```
## 3.	Creación del entorno virtual
Se recomienda trabajar este proyecto en un entorno virtual para evitar conflictos con otras instalaciones de Python. Para ello, ejecute el siguiente comando en la terminal:
```
python -m venv venv
```
Luego, puede proceder a activar el entorno, dependiendo de su sistema operativo, con el siguiente comando:

**•	Windows**
```
venv\Scripts\activate
```

**•	macOS / Linux**
```
source venv/bin/activate
```
## 4.	Instalación de dependencias
El proyecto incluye un archivo requirements.txt con todas las librerías necesarias para ejecutar el modelo y el dashboard. Para instalar las dependencias de este archivo, ejecute el siguiente comando:
```
pip install -r requirements.txt
```
Esto instalará librerías como: pandas, numpy, scikit-learn, mlflow, dash / plotly, joblib; entre otras requeridas por el sistema.
## 5.	Estructura del proyecto
Una vez descargado el proyecto, encontrará las siguientes carpetas principales:

•	data/ : Contiene los archivos train.csv y test.csv.

•	dashboard/ :  Incluye la aplicación interactiva (app.py).

•	models/ : Carpeta donde se almacenan modelos entrenados y artefactos.

•	notebooks/ :  Jupyter notebooks utilizados en el desarrollo.

•	docs/ : Manuales y documentación.

•	api/ : Código base del API si se desea exponer el modelo.

## 6.	Ejecución del modelo
Para que el modelo sea entrenado, necesita correr en su máquina o máquina virtual el archivo `python train_rf.py` que se encuentra en la carpeta models, esto generará un joblib en models/artifacts. Este archivo deberá ser copiado a la carpeta api para que tanto la API como el dashboard utilicen el modelo escogido.

Si ya tiene un modelo entrenado, puede poner el joblib directamente en la carpeta api y usar inmediatamente el dashboard.

## 7.	Ejecución del dashboard interactivo
El proyecto incluye un tablero construido con Dash que permite visualizar resultados y usar el modelo para predicciones. Para saber cómo manejar esta herramienta puede dirigirse al manual_usuario de la carpeta docs/, pero para correrlo, debe ejecutar el siguiente comando en su terminal:
```
cd dashboard
python app.py
```
Luego abra en su navegador la ruta para poder visualizar e interactuar con el dashboard:
```
http://localhost:8050
```
