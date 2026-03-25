import streamlit as st
import json
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
from catboost import CatBoostRegressor
import openai
from openai import OpenAI


@st.cache_resource
def load_model():

    encoder = SentenceTransformer('intfloat/multilingual-e5-base')


    index_md = faiss.read_index('vector_bd/ideal_names_e5.index')
    vector_bd_paths = pd.read_csv('vector_bd/ideal_names_list.csv').iloc[:, 0].tolist()

    catboost_model = CatBoostRegressor()
    catboost_model.load_model('model_catboost/catboost_2.cbm')


    index_md2 = faiss.read_index('vector_bd/ideal_names_e52.index')
    vector_bd_paths2 = pd.read_csv('vector_bd/ideal_names_list2.csv').iloc[:, 0].tolist()

    catboost_model2 = CatBoostRegressor()

    catboost_model2.load_model('model_catboost/catboost_car_repair_model2.cbm')

    return encoder, index_md, vector_bd_paths, catboost_model, index_md2, vector_bd_paths2, catboost_model2



encoder, index_md, vector_bd_paths, catboost_model, index_md2, vector_bd_paths2, catboost_model2 = load_model()
if "parsed_json" not in st.session_state:
    st.session_state.parsed_json = None
if "works_five" not in st.session_state:
    st.session_state.works_five = None
if "works_five_2" not in st.session_state:
    st.session_state.works_five_2 = None

st.set_page_config(page_title='DEMO', page_icon='🚗', layout='wide')
st.title('DEMO APP 🤖')
st.write("""
    Оценка нормо-часов.
""")

try:
    api_key = st.secrets['GROQ_API_KEY']
    client = OpenAI(
        base_url='https://api.groq.com/openai/v1',
        api_key=api_key
    )
except KeyError:
    st.warning(KeyError)
    st.stop()

slang_dictionary = """
    СЛОВАРЬ АВТОСЛЕНГА ДЛЯ ПЕРЕВОДА:
    - Автомат, Тяпка -> автоматическая коробка передач
    - Банан, Желток, Таблетка -> запасное колесо
    - Банка, Колокол, Пердулина, Хор-хор -> глушитель
    - Весла, Мясорубка -> ручные стеклоподъемники
    - Гидрач -> гидроусилитель руля
    - Голова -> головка блока цилиндров
    - Горшок, Котел -> цилиндр двигателя
    - Граната -> ШРУС
    - Гриль -> решетка радиатора
    - Губа -> спойлер бампера
    - Движок -> двигатель
    - Домик, Краб -> кронштейн
    - Зенки -> фары
    - Зерцало, Уши, Рога -> зеркало заднего вида
    - Кенгурятник -> защита бампера
    - Колено -> коленчатый вал
    - Кондей -> компрессор кондиционера
    - Кочерга, Палка, Мешалка, Мотыга, Ручка -> МКПП
    - Метла, Хвост -> задний стеклоочиститель
    - Мослы -> стеклоочистители ветрового стекла
    - Муфта, Ступление -> сцепление
    - Мухобойка -> дефлектор капота
    - Рука дружбы -> шатун двигателя
    - Ручник, Якорь -> стояночный тормоз
    - Сигналка, Клю-Клю, Чвакалка -> автосигнализация
    - Удочка -> антенна
    - Феродо -> диск сцепления
    - Ходовка -> подвеска
    - Шар -> подушка безопасности
    - Шарманка, Патефон -> автомагнитола
    """

system_prompt = f"""
Ты суровый автомеханик-диагност. Твоя задача — извлечь марку, модель и перевести жалобу клиента на строгий технический язык.
ПРАВИЛА:
1. ЗАЩИТА ОТ БРЕДА: Если просят абсурд, верни: {{"brand": "error", "model": "error", "work": "error"}}.
2. АНГЛИЙСКИЙ ДЛЯ БРЕНДОВ: ВСЕГДА переводи марки и модели на английский! ("Volkswagen", "Polo").
3. СЛОВАРЬ: {slang_dictionary}
4. НЕИЗВЕСТНЫЕ АВТО: Если марка не указана, пиши "unknown".
5. СУТЬ РАБОТЫ: В поле "work" напиши ПОЛНУЮ техническую фразу (деталь + действие) СТРОГО НА РУССКОМ ЯЗЫКЕ. 
   Пример 1: "хрустит граната" -> "шарнир равных угловых скоростей замена".
   Пример 2: "поменять движок" -> "двигатель снятие/установка".
6. ТИП ДВИГАТЕЛЯ (КРИТИЧНО): Определи тип двигателя. Пиши строго одно из двух:
   * "electric" (если это Zeekr, Tesla, Nio, Voyah FREE, Lixiang, или если в запросе есть "электромобиль", "электричка", "ВВБ")
   * "ice" (Internal Combustion Engine - для всех остальных бензиновых/дизельных машин)
7. КОЭФФИЦИЕНТ СЛОЖНОСТИ (multiplier): Оцени марку автомобиля и выдай число (float):
   - 1.0 : Простые и массовые авто (Например, Lada, Hyundai, Kia, Renault, Toyota, Volkswagen).
   - 1.5 : Премиум и сложный ремонт (Например, Audi, BMW, Mercedes, Porsche).
   - 2.5 : Электромобили, редкие китайцы и суперкары (Например, Zeekr, Tesla, Lixiang, Voyah).
   - Если марка unknown, ставь 1.0.
Ответь ТОЛЬКО JSON: {{"brand": "...", "model": "...", "work": "...", "engine_type": "...", "multiplier": 1.0}}
"""

with st.form("form"):
    user_input = st.text_area(
        "Введите запрос",
        placeholder="Например: Ауди А6, сломался двигатель"
    )
    submit_button = st.form_submit_button('Поиск')

    if submit_button and user_input.strip():
        with st.spinner("Диагностирую..."):
            try:
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0.1,
                    timeout=15
                )
            except openai.APITimeoutError:
                st.error("К сожалению, тестовый сервер LLM-провайдера сейчас недоступен. Пожалуйста, нажмите 'Поиск' еще раз.")
            except Exception as e:
                st.error(f"Произошла ошибка при обращении к нейросети. Детали: {e}")
            llm_answer = resp.choices[0].message.content.strip()
            llm_answer = llm_answer.replace("```json", "").replace("```", "").strip()

            try:
                parsed = json.loads(llm_answer)
                if parsed.get('brand') == 'error':
                    st.error("Запрос не содержит модель или название работы")
                else:
                    st.session_state.parsed_json = parsed
                    work_query = parsed.get('work', '')
                    query_text = f'query: {work_query}'
                    query_vector = encoder.encode([query_text], normalize_embeddings=True)

                    # --- Поиск по первому индексу (Старая база) ---
                    distance1, indices1 = index_md.search(query_vector, k=5)
                    st.session_state.works_five = [vector_bd_paths[i] for i in indices1[0]]

                    # --- Поиск по второму индексу (Новая база / 95-й процентиль) ---
                    distance2, indices2 = index_md2.search(query_vector, k=5)
                    st.session_state.works_five_2 = [vector_bd_paths2[i] for i in indices2[0]]
            except json.JSONDecodeError:
                st.error("Ошибка парсинга JSON от Llama.")
                st.write(llm_answer)

if st.session_state.parsed_json and st.session_state.works_five and st.session_state.works_five_2:
    parsed = st.session_state.parsed_json
    works_five = st.session_state.works_five
    works_five_2=st.session_state.works_five_2
    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        st.success("Ответ LLama")
        st.json(parsed)

    with col2:
        st.info("Версия 1")
        selected_ideal_work = st.selectbox(
            "FAISS нашел варианты. Выберите точный:",
            works_five
        )

        df_pred = pd.DataFrame([{
            'brand': 'unknown',
            'model': 'unknown',
            'mode': 'снятие/установка',
            'ideal_name': selected_ideal_work
        }])

        base_time_hours = catboost_model.predict(df_pred)[0]
        base_time_hours = max(0.1, base_time_hours)

        multiplier = float(parsed.get('multiplier', 1.0))
        time_hours = base_time_hours * multiplier

        time_mins = int(time_hours * 60)

        st.divider()

        if multiplier > 1.0:
            st.warning(
                f"Для марки {parsed.get('brand')} применен поправочный коэффициент сложности (x{multiplier}).")

        st.metric(
            label="ИТОГОВЫЙ ПРОГНОЗ ВРЕМЕНИ",
            value=f"{time_hours:.2f} нормо-часа",
            delta=f"{time_mins} минут",
            delta_color="off"
        )
    with col3:
        st.info("Версия 2")
        selected_ideal_work_2 = st.selectbox(
            "FAISS нашел варианты. Выберите точный:",
            works_five_2,
            key="v2_select"
        )


        df_pred_v2 = pd.DataFrame([{
            'brand': str(parsed.get('brand')).lower(),
            'model': str(parsed.get('model')).lower(),
            'car_class': 'unknown',
            'work_category': 'Прочее',
            'work_name_clean': selected_ideal_work_2
        }])


        base_time_hours2 = catboost_model2.predict(df_pred_v2)[0]
        base_time_hours2 = max(0.1, base_time_hours2)

        multiplier = float(parsed.get('multiplier', 1.0))
        time_hours2 = base_time_hours2 * multiplier

        time_mins2 = int(time_hours2 * 60)

        st.divider()

        if multiplier > 1.0:
            st.warning(
                f"Для марки {parsed.get('brand')} применен поправочный коэффициент сложности (x{multiplier}).")

        st.metric(
            label="ИТОГОВЫЙ ПРОГНОЗ ВРЕМЕНИ",
            value=f"{time_hours2:.2f} нормо-часа",
            delta=f"{time_mins2} минут",
            delta_color="off"
        )
