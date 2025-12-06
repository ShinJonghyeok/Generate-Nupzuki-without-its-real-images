from PIL import Image

# 불러올 이미지 파일 경로
img_paths = [
    "final/0.png",
    "final/1.png",
    "final/2.png",
    "final/3.png"
]

canvas = Image.new("RGB", (2048, 2048))

# 파일 읽고 배치
positions = [
    (0, 0),          # top-left
    (1024, 0),       # top-right
    (0, 1024),       # bottom-left
    (1024, 1024)     # bottom-right
]

for path, pos in zip(img_paths, positions):
    img = Image.open(path)
    # 이미지가 1024x1024라는 전제지만, 혹시 다르면 resize 해줌
    img = img.resize((1024, 1024))
    canvas.paste(img, pos)

# 저장
canvas.save("team22_final_content.png", "PNG")
print("saved: team22_final_content.png")