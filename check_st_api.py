
import streamlit
try:
    from streamlit.elements import image as st_image
    print(f"Has image_to_url: {hasattr(st_image, 'image_to_url')}")
    if hasattr(st_image, 'image_to_url'):
        print(st_image.image_to_url)
except Exception as e:
    print(e)
