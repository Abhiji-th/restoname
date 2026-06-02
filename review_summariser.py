from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated, Optional, Literal 
from dotenv import load_dotenv

load_dotenv()

class review(TypedDict):
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["Pos", "Neg"], "Sentiment of the review either Pos (for positive) and Neg (for negative)"]
    pros: Annotated[Optional[list[str]], "Pros in the review"]
    cons: Annotated[Optional[list[str]], "Cons in the review"]
    word_length: Annotated[int, "Total number of words in the review"]

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct"
)

model = ChatHuggingFace(llm=llm)

strutured_model = model.with_structured_output(review)

query = """
**Review: Samsung S26 Ultra**

I've been using the Samsung S26 Ultra for about three weeks, and overall it has been a great experience. The display is stunning, with vibrant colors, excellent brightness, and smooth scrolling thanks to the high refresh rate. Whether I'm watching videos, browsing social media, or editing photos, the screen quality is among the best I've seen on a smartphone.

The camera system is impressive. Daylight photos are sharp and detailed, while low-light performance has improved noticeably compared to previous models. The zoom capabilities are especially useful for capturing distant subjects without losing too much detail. Battery life is also excellent; with moderate to heavy usage, I can comfortably get through a full day without needing to recharge.

Performance is another strong point. Apps open quickly, multitasking is seamless, and gaming performance remains smooth even during extended sessions. Samsung's software offers many customization options, which power users may appreciate.

However, there are a few drawbacks. The phone is quite large and heavy, making one-handed use difficult. While the cameras are generally excellent, image processing can sometimes oversaturate colors, making photos look less natural. The device is also expensive, and some users may find it difficult to justify the premium price. Additionally, the charging speed, although decent, still lags behind some competitors in the flagship segment.

### Pros

* Exceptional display quality with high brightness and smooth refresh rate
* Excellent camera performance, especially zoom and low-light photography
* Long-lasting battery life
* Fast and reliable performance for multitasking and gaming
* Premium build quality and design
* Rich customization features in the software

### Cons

* Expensive compared to many competitors
* Large and heavy, making one-handed use challenging
* Camera processing can sometimes produce overly saturated colors
* Charging speeds are not the fastest in the flagship market
* Premium accessories may need to be purchased separately

### Overall Rating

**4.5/5**

The Samsung S26 Ultra is an outstanding flagship smartphone with excellent cameras, display quality, and battery life. While its size, weight, and price may not suit everyone, it remains a strong choice for users looking for a premium Android experience.

"""

result = strutured_model.invoke(query)

print(result)