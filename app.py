import os
from dotenv import load_dotenv
import streamlit as st
import matplotlib.pyplot as plt

from crewai import Agent, Task, Crew, LLM

# -----------------------------
# CONFIG
# -----------------------------
load_dotenv()

st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="🌍",
    layout="wide"
)

# -----------------------------
# GEMINI LLM (STABLE)
# -----------------------------
llm = LLM(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# -----------------------------
# HEADER
# -----------------------------
st.title("🌍 AI Travel Planner")
st.caption("Plan smart trips with AI agents ✈️")

# -----------------------------
# SIDEBAR MEMORY
# -----------------------------
st.sidebar.header("🧠 Preferences")

if "preferences" not in st.session_state:
    st.session_state.preferences = ""

preferences = st.sidebar.text_area(
    "Your travel style (e.g. veg food, beaches, nightlife):",
    value=st.session_state.preferences
)

if st.sidebar.button("💾 Save Preferences"):
    st.session_state.preferences = preferences
    st.sidebar.success("Saved!")

# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader("📋 Trip Details")

col1, col2, col3 = st.columns(3)

with col1:
    destination = st.text_input("📍 Destination")

with col2:
    days = st.number_input("📅 Days", 1, 15, 3)

with col3:
    budget = st.number_input("💰 Budget (INR)", 1000, 1000000, 20000)

travel_type = st.selectbox(
    "🧳 Travel Type",
    ["budget", "luxury", "family", "solo"]
)

# -----------------------------
# BUTTON
# -----------------------------
if st.button("🚀 Generate Travel Plan", use_container_width=True):

    if not destination:
        st.error("⚠️ Please enter a destination")
        st.stop()

    with st.spinner("🤖 AI agents are planning your trip..."):

        # -----------------------------
        # AGENTS
        # -----------------------------
        planner = Agent(
            role="Travel Planner",
            goal=f"Create a structured {days}-day itinerary for {destination}",
            backstory="Expert travel planner with focus on optimization",
            llm=llm
        )

        budget_agent = Agent(
            role="Budget Analyst",
            goal="Provide accurate cost breakdown",
            backstory="Expert in travel financial planning",
            llm=llm
        )

        guide = Agent(
            role="Local Guide",
            goal="Suggest best experiences and hidden gems",
            backstory="Knows culture, food, and attractions",
            llm=llm
        )

        personalizer = Agent(
            role="Travel Personalizer",
            goal="Customize plan based on user type",
            backstory="Expert in personalized travel experiences",
            llm=llm
        )

        # -----------------------------
        # TASKS (STRUCTURED OUTPUT)
        # -----------------------------
        plan_task = Task(
            description=f"""
            Create a {days}-day travel itinerary for {destination}.
            Preferences: {preferences}

            Format:
            Day 1:
            - Morning:
            - Afternoon:
            - Evening:
            """,
            expected_output="Structured day-wise itinerary",
            agent=planner
        )

        budget_task = Task(
            description=f"""
            Provide budget breakdown within {budget} INR.
            """,
            expected_output="""
            Format:
            Transport: ₹
            Hotel: ₹
            Food: ₹
            Activities: ₹
            Total: ₹
            """,
            agent=budget_agent
        )

        guide_task = Task(
            description=f"""
            Suggest top attractions and food in {destination}.
            """,
            expected_output="""
            - Top Places
            - Food Recommendations
            - Hidden Gems
            """,
            agent=guide
        )

        personalize_task = Task(
            description=f"""
            Adjust plan for {travel_type} traveler.
            Preferences: {preferences}
            """,
            expected_output="Customized itinerary improvements",
            agent=personalizer
        )

        # -----------------------------
        # CREW
        # -----------------------------
        crew = Crew(
            agents=[planner, budget_agent, guide, personalizer],
            tasks=[plan_task, budget_task, guide_task, personalize_task],
            verbose=False
        )

        result = crew.kickoff()

    # -----------------------------
    # OUTPUT SECTION
    # -----------------------------
    st.success("✅ Your Travel Plan is Ready!")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🧾 AI Generated Plan")
        st.markdown(result)

    with col2:
        st.subheader("📊 Budget Split")

        transport = budget * 0.3
        hotel = budget * 0.4
        food = budget * 0.2
        activities = budget * 0.1

        labels = ["Transport", "Hotel", "Food", "Activities"]
        values = [transport, hotel, food, activities]

        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct="%1.1f%%")
        ax.set_title("Budget Allocation")

        st.pyplot(fig)

    # -----------------------------
    # SMART INSIGHTS
    # -----------------------------
    st.subheader("💡 AI Tips")

    tips = {
        "budget": "Use local transport & budget stays.",
        "luxury": "Go for premium hotels & curated experiences.",
        "family": "Choose safe & kid-friendly places.",
        "solo": "Explore hostels & meet local communities."
    }

    st.info(tips[travel_type])