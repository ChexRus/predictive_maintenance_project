import streamlit as st
from analysis_and_model import analysis_and_model_page 
from presentation import presentation_page 

# Настройка навигации
pages = {
    "Анализ и модель": analysis_and_model_page,
    "Презентация": presentation_page,
}

# Отображение навигации
current_page = st.sidebar.selectbox("Выберите страницу", options=list(pages.keys()))

# Вызов выбранной страницы
pages[current_page]()
