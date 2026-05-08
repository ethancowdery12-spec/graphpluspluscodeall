import modal
app = modal.App("test-tokenizers")
image = modal.Image.debian_slim(python_version="3.11").pip_install("transformers==4.46.0", "tokenizers")
@app.function(image=image)
def test_imports():
    import tokenizers
    print("SUCCESS")
