from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
        You are a travel guide.

        Suggest a {trip_type} trip to {destination} for a duration of {days} days.
        Include:
        - Top attractions
        - Recommended food
        - Estimated budget
        """,
    input_variables=['trip_type', 'destination', 'days']
)

template.save("template.json")
