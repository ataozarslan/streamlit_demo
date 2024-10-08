import streamlit as st
import pandas as pd

# Sayfa Ayarları
st.set_page_config(
    page_title="Hotel Cancellations",
    page_icon="images/hotel-service.png",
    layout="wide",
    menu_items={
        "Get help": "mailto:ata.ozarslan@istdsa.com",
        "About": "For More Information\n" + "https://istdatascience.com/"
    }
)

st.title("About Data")
st.markdown("Here is our dataset and data dictionary belongs to our hotel!")

st.header("Inspect The Data")

df = pd.read_csv("hotel_bookings_prepared.csv")
booking_status = st.selectbox("**Booking Status Selection**", df["booking_status"].unique())
filtered_df = df[df["booking_status"] == booking_status]
st.dataframe(filtered_df.sample(10), hide_index=True)

st.header("Data Dictionary")

st.markdown("""
| Column     | Description              |
|------------|--------------------------|
| `Booking_ID` | Unique identifier of the booking. |
| `no_of_adults` | The number of adults. |
| `no_of_children` | The number of children. |
| `no_of_weekend_nights` | Number of weekend nights (Saturday or Sunday). |
| `no_of_week_nights` | Number of week nights (Monday to Friday). |
| `type_of_meal_plan` | Type of meal plan included in the booking. |
| `required_car_parking_space` | Whether a car parking space is required. |
| `room_type_reserved` | The type of room reserved. |
| `lead_time` | Number of days before the arrival date the booking was made. |
| `arrival_year` | Year of arrival. |
| `arrival_month` | Month of arrival. |
| `arrival_date` | Date of the month for arrival. |
| `market_segment_type` | How the booking was made. |
| `repeated_guest` | Whether the guest has previously stayed at the hotel. |
| `no_of_previous_cancellations` | Number of previous cancellations. |
| `no_of_previous_bookings_not_canceled` | Number of previous bookings that were canceled. |
| `avg_price_per_room` | Average price per day of the booking. |
| `no_of_special_requests` | Count of special requests made as part of the booking. |
| `booking_status` | Whether the booking was cancelled or not. |
""")