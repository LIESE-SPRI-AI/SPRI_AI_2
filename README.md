# WildFire C

---

# Requisitos

Antes de compilar el proyecto debes tener instalado:

- CMake
- Ninja
- Doxygen *(opcional, para documentación)*

Verificar herramientas:

```bash
cmake --version
ninja --version
doxygen -v
```

---

# Compilación del proyecto

## 1. Crear carpeta de compilación

```bash
mkdir build
cd build
```

## 2. Generar archivos de compilación

```bash
cmake -G Ninja ..
```

## 3. Compilar

```bash
cmake --build . 

```

## 4. Ejecutar

```bash
./app.exe   

```

Los archivos generados aparecerán dentro del directorio `build/`.

---

# Generar documentación

Desde la raíz del proyecto ejecutar:

```bash
doxygen Doxyfile
```

La documentación HTML principal se genera en:

```txt
docs/html/index.html
```
---

# Estructura del proyecto

```txt
project/
│
├── include/          # Headers
├── src/              # Implementaciones
├── docs/             # Documentación generada
│   └── html/
├── build/            # Archivos compilados
├── CMakeLists.txt
├── Doxyfile
└── README.md
```
---


