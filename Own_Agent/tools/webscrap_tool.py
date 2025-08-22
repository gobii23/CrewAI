import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
import re
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool

class WebScraperTool(BaseTool):
    name: str = "WebScraperTool"
    description: str = (
        "Takes an input Excel file with columns 'School' and 'State/UT', "
        "searches for each school's official website, scrapes it for details "
        "(District, Address, Tel, Email, Website), and saves a new enriched Excel file."
    )

    def _run(self, input_excel: str, output_excel: str = "enriched_schools.xlsx") -> str:
        try:
            df = pd.read_excel(input_excel)
            search_tool = SerperDevTool()

            enriched_rows = []

            for _, row in df.iterrows():
                query = f"{row['School']} {row['State/UT']}"
                
                # Step 1: Get website with Serper
                website = search_tool.run(query)
                if not website:
                    enriched_rows.append({
                        **row.to_dict(),
                        "District": None, "Address": None,
                        "Tel": None, "Email": None, "Website": None
                    })
                    continue

                # Step 2: Scrape website
                scraped_data = self._scrape_website(website)

                enriched_rows.append({
                    **row.to_dict(),
                    "District": scraped_data.get("district"),
                    "Address": scraped_data.get("address"),
                    "Tel": scraped_data.get("telephone"),
                    "Email": scraped_data.get("email"),
                    "Website": scraped_data.get("website")
                })

            # Save enriched Excel
            enriched_df = pd.DataFrame(enriched_rows)
            enriched_df.to_excel(output_excel, index=False)
            return output_excel

        except Exception as e:
            return f"❌ Error during enrichment: {str(e)}"

    def _scrape_website(self, website_url: str) -> dict:
        """Scrape contact details from a website."""
        try:
            response = requests.get(website_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(" ", strip=True)

            email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
            tel_match = re.search(r"\+?\d[\d\s-]{7,15}", text)
            address_match = re.search(r"(Address|Addr|Location)[:\-]?\s(.+?)(\n|$)", text, re.IGNORECASE)

            return {
                "district": None,  # could add smarter extraction
                "address": address_match.group(2).strip() if address_match else None,
                "telephone": tel_match.group(0) if tel_match else None,
                "email": email_match.group(0) if email_match else None,
                "website": website_url
            }

        except Exception as e:
            return {"error": str(e), "website": website_url}
