from PIL import ImageDraw, ImageFont, Image
from copy import deepcopy


def draw_point_on_image(image, point, val, color=(255, 0, 0), radius=None, font=None):
    """
    image: PIL.Image (RGB/RGBA)
    point: (x_norm, y_norm)  # 0~1 정규화 좌표
    val:   float              # 포인트 값(예: 0~1)
    color: 점/텍스트 색상
    radius: 점 반지름(px). None이면 이미지 크기에 비례해 자동 설정
    font:  PIL.ImageFont. None이면 기본 폰트 사용
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image")

    img = image
    draw = ImageDraw.Draw(img)

    W, H = img.size
    x = int(round(point[0] * (W - 1)))
    y = int(round(point[1] * (H - 1)))

    if radius is None:
        radius = max(2, int(min(W, H) * 0.008))  # 이미지 크기 비례 (대략 0.8%)

    # 점(원) 그리기
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, fill=color, outline=color, width=1)

    # 텍스트(값) 그리기
    if font is None:
        try:
            # 시스템에 따라 다를 수 있어 예외 처리
            font = ImageFont.truetype("arial.ttf", max(12, int(radius * 2)))
        except Exception:
            font = ImageFont.load_default()

    text = f"{val:.3f}"
    # 텍스트가 점을 가리지 않도록 약간 오른쪽 위로 오프셋
    tx = min(max(0, x + radius + 2), W - 1)
    ty = min(max(0, y - radius - 2), H - 1)

    # 가독성을 위한 외곽선(stroke)
    try:
        draw.text((tx, ty), text, fill=color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    except TypeError:
        # PIL 구버전 호환
        draw.text((tx, ty), text, fill=color, font=font)

    return img


def draw_points_on_image(image, points, vals, color=(255, 0, 0), radius=None, font=None, colors=None):
    """
    image:  PIL.Image
    points: [(x_norm, y_norm), ...]  # 0~1 정규화 좌표
    vals:   [float, ...]             # 각 포인트의 값 (텍스트로 표시)
    color:  모든 포인트에 동일 색을 쓰고 싶을 때
    radius: 점 반지름(px). None이면 자동
    font:   PIL.ImageFont
    colors: 포인트별 색을 개별 지정하고 싶을 때. len(colors) == len(points)
    """
    if points is None or len(points) == 0:
        return image

    if vals is None or len(vals) != len(points):
        raise ValueError("`vals`의 길이는 `points`와 같아야 합니다.")

    img = deepcopy(image)
    use_per_point_color = colors is not None and len(colors) == len(points)

    for idx, (p, v) in enumerate(zip(points, vals)):
        c = colors[idx] if use_per_point_color else color
        img = draw_point_on_image(img, p, v, color=c, radius=radius, font=font)
    return img

