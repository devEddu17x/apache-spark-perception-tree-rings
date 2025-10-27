#!/bin/bash

# ========================================
# 🧰 Proyecto: Setup automático
# Configura el entorno de pre-commit y linters
# ========================================

echo "🚀 Starting environment setup..."

# Verificar que pip esté instalado
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install it before continuing."
    exit 1
fi

# Crear entorno virtual si no existe
if [ ! -d "env" ]; then
    echo "🐍 Creating virtual environment..."
    python -m venv env
else
    echo "✅ Existing virtual environment detected."
fi

# Activar entorno virtual
echo "🔗 Activating virtual environment..."
source env/bin/activate

# Instalar dependencias del proyecto (si existe requirements.txt)
if [ -f "requirements.txt" ]; then
    echo "📦 Installing project dependencies..."
    pip install -r requirements.txt
fi

# Instalar pre-commit
echo "⚙️ Installing pre-commit..."
pip install pre-commit

# Instalar hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# (Opcional) Actualizar hooks a la última versión permitida
pre-commit autoupdate

echo ""
echo "✅ Setup completed successfully."

