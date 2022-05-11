import requests
import streamlit as st
# Parameters

st.markdown('''## 🌤 **App To Give You The Weather Details** ☀️''')

city = st.text_input('Enter A City Name', '<e.g. Chicago>')

if city !='<e.g. Chicago>' or city !=' ':
    unit = 'imperial'
    url = 'http://api.openweathermap.org/data/2.5/weather'

    # Fetch API key from file
    file_input = open('/Users/alf/PycharmProjects/pythonProject/weather_api_key', 'rb')
    for file in file_input:
        api_key = file
    #
    # city = input('Enter A City Name:')
    #
    try:
        response = requests.get(f'{url}?q={city}&appid={api_key.decode()}&units={unit}')
    except(requests.RequestException) as error:
        print(error)
    finally:
        if response.status_code == 200:
            web_data = response.json()
            # (web_data)
            if 'cloud' in web_data['weather'][0]['description']:
                st.write(f"### *Temperature Of {city.title()} Is: {web_data['main']['temp']} °F ☁️*")
                st.write(f"### *Condition For {city.title()} Looks Like: {web_data['weather'][0]['description'].title()} ☁️*")
            elif 'sun' in web_data['weather'][0]['description']:
                st.write(f"### *Temperature Of {city.title()} Is: {web_data['main']['temp']} °F ☀️*")
                st.write(f"### *Condition For {city.title()} Looks Like: {web_data['weather'][0]['description'].title()} ☀️*")
            elif 'rain' in web_data['weather'][0]['description']:
                st.write(f"### *Temperature Of {city.title()} Is: {web_data['main']['temp']} °F 🌧️*")
                st.write(f"### *Condition For {city.title()} Looks Like: {web_data['weather'][0]['description'].title()} 🌧️*")
            elif 'sky' in web_data['weather'][0]['description']:
                st.write(f"### *Temperature Of {city.title()} Is: {web_data['main']['temp']} °F 🌝*")
                st.write(f"### *Condition For {city.title()} Looks Like: {web_data['weather'][0]['description'].title()} 🌝*")
            else:
                st.write(f"### *Temperature Of {city.title()} Is: {web_data['main']['temp']} °F*")
                st.write(f"### *Condition For {city.title()} Looks Like: {web_data['weather'][0]['description'].title()}*")
        else:
            # st.write(f'### *{response.status_code} error code: Sorry Enter A Valid City Name*')
            st.write(f'### *Please Enter A Valid City Name*')