from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ice_breaker import ice_break
from pydantic import BaseModel, Field

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:5174",  # in case of frontend failure
    "http://localhost:4173",  # build application
    "http://localhost:4174",  # build application fallback
    "http://localhost:3000",  # react dev server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserForm(BaseModel):
    short_name: str = Field(description="First name of the person")


@app.post("/icebreaker")
async def generate_icebraker(user_form: UserForm):
    person_info, profile_pic_url, short_name = ice_break(
        short_name=user_form.short_name, mode="local"
    )

    return {
        "short_name": short_name,
        "summary": person_info.summary,
        "interests": person_info.topics_of_interest,
        "facts": person_info.facts,
        "ice_breakers": person_info.ice_breakers,
        "picture_url": profile_pic_url,
    }
