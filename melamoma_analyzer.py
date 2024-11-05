import os

from ourmodel import FinalModel

import streamlit as st
from streamlit_echarts import st_echarts
from PIL import Image

def update_and_save_img(img_file_buffer) -> None:
    if "img.jpg" in os.listdir():
        os.remove("img.jpg")
    
    try:
        img = Image.open(img_file_buffer)
        img.save("img.jpg")

    except AttributeError as error:
        return error

def draw_pie(data):
    options = {
        "tooltip": {"trigger": "item"},
        "legend": {"top": "5%", "left": "center"},
        "series": [
            {
                "name": "pie",
                "type": "pie",
                "radius": ["40%", "70%"],
                "avoidLabelOverlap": False,
                "itemStyle": {
                    "borderRadius": 10,
                    "borderColor": "#fff",
                    "borderWidth": 2,
                },
                "label": {"show": False, "position": "center"},
                "emphasis": {
                    "label": {"show": True, "fontSize": "40", "fontWeight": "bold"}
                },
                "labelLine": {"show": False},
                "data": [],
            }
        ],
    }

    options["series"][0]["data"] += data

    st_echarts(
        options=options, height="250px",
    )

st.title("Skin cancer analyzer", anchor=False)
st.subheader('Developed by "Gosling team"')
st.divider()

col_1, col_2 = st.columns(2)

with col_1:
    st.subheader(":blue[О приложении]")
    st.write("Приложение оценивает вероятность развития меланомы по фотографии кожных покровов.")

with col_2:
    st.subheader(":blue[Как сохранить результат?]")
    st.write("В правом верхнем углу выберете пункт 'Print', чтобы распечатать страницу, или воспользуйтесь сочетанием 'ctrl + s', чтобы сохранить страницу.")

st.divider()
img_file_buffer = st.file_uploader(":blue[Загрузите фотографию в формате .jpg]", type='jpg')
# img_file_buffer = st.camera_input(":blue[Нажмите на кнопку, чтобы снять фото]", label_visibility='visible', disabled=False)

col_3, col_4 = st.columns(2)
model = FinalModel(6)

if img_file_buffer is not None:
    try:
        with st.spinner('Обработка изображения…'):
            update_and_save_img(img_file_buffer)

        with col_3:
            prediction = model.calibrated_predict("img.jpg")
            st.subheader(":blue[Заключение]")
            st.write(f"Результат: Вероятность злокачественности на фото: {prediction['binary_probs'][0] * 100:.1f}%. \nДанное фото имеет наибольшее сходство с {prediction['multiclass_preds'][0]}")

        with col_4:
            st.subheader(":blue[Обработанная фотография]")
            st.image(img_file_buffer)

    except TypeError as e:
        st.warning("Объект не найден, загрузите другую фотографию!", icon="🚨")
