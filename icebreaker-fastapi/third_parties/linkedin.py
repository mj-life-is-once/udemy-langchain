# fetch and scrape LinkedIn information
import json
import os

import requests


# langchain is going to use docstring to decide whether or not to
# use a specific function
def scrape_linkedin_profile(
    linkedin_profile_url: str, save_to_file: bool = False, name: str = "result"
):
    """scrape information from LinkedIn profiles,
    Manually scrape the information from the LinkedIn profile"""
    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    header_dic = {"Authorization": f'Bearer {os.environ["PROXY_CURL_API_KEY"]}'}

    response = requests.get(
        api_endpoint, params={"url": linkedin_profile_url}, headers=header_dic
    )
    data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    if save_to_file:
        with open(f"local/{name}.json", "w") as f:
            json.dump(data, f)
    return data
