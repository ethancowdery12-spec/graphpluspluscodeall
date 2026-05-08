import os
import modal

app = modal.App("check-vol")
vol = modal.Volume.from_name("graphrag-finetune-vol")

@app.function(volumes={"/data": vol})
def check_files():
    print("Files in /data:")
    for root, dirs, files in os.walk("/data"):
        for name in files:
            p = os.path.join(root, name)
            size = os.path.getsize(p)
            print(f"{p}: {size / 1e9:.2f} GB")
        for name in dirs:
            print(f"DIR: {os.path.join(root, name)}")

if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            check_files.remote()
