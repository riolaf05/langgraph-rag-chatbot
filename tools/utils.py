from langchain.agents import Tool

def get_today_date(input : str) -> str:
    import datetime
    today = datetime.date.today()
    return f"\n {today} \n"

def get_summarized_text(text : str) -> str:
    from transformers import pipeline
    summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    return summarizer(text, max_length=1000, min_length=30, do_sample=False)[0]['summary_text']

get_today_date_tool = Tool(
    name="Ottieni data",
    func=get_today_date,
    description="Useful for getting today's date"
)

get_summarized_text_tool = Tool(
    name="Get Summarized Text",
    func=get_summarized_text,
    description="Useful for getting summarized text for any document."
)