from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import fitz  # pymupdf
import base64
import requests
import pandas as pd
from io import StringIO
from sqlmodel import SQLModel, Field, create_engine, Session, select
from typing import Optional

# --- Config ---
OPENROUTER_API_KEY = "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"
VISION_MODEL = "mistralai/mistral-small-3.2-24b-instruct:free"
IMAGE_DIR = "pdf_images"
DB_PATH = "results.db"

os.makedirs(IMAGE_DIR, exist_ok=True)

# --- Schema (same as before) ---
class Constituency(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str

class Mandal(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    constituency_id: int = Field(foreign_key="constituency.id")
    name: str

class Village(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    constituency_id: int = Field(foreign_key="constituency.id")
    mandal_id: int = Field(foreign_key="mandal.id")
    name: str

class PollingStation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    village_id: int = Field(foreign_key="village.id")
    year: int

class Party(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str

class Candidate(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    party_id: int = Field(foreign_key="party.id")
    constituency_id: int = Field(foreign_key="constituency.id")
    name: str

class VoteCount(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    polling_station_id: int = Field(foreign_key="pollingstation.id")
    candidate_id: int = Field(foreign_key="candidate.id")
    votes: int

engine = create_engine(f"sqlite:///{DB_PATH}")
SQLModel.metadata.create_all(engine)

DUMMY_CONSTITUENCY = "Sample Constituency"
DUMMY_MANDAL = "Sample Mandal"
DUMMY_VILLAGE = "Sample Village"
DUMMY_YEAR = 2024

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def convert_pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    image_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        image_path = os.path.join(IMAGE_DIR, f"page_{i + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
    return image_paths

def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_prompt(image_base64):
    return {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract election results data from this image and return ONLY a CSV with these exact columns:\\n\\n"
                            "polling_station,All India Anna Dravida Munnetra Kazhagam,Dravida Munnetra Kazhagam,"
                            "Naam Tamilar Katchi,Makkal Needhi Maiam,Bharatiya Janata Party,Independent,NOTA,total\\n\\n"
                            "Rules:\\n"
                            "- Return ONLY the CSV data, no explanations or text\\n"
                            "- Include only polling station summary rows\\n"
                            "- Exclude individual candidate name rows\\n"
                            "- Ensure exactly 9 columns as specified\\n"
                            "- Use comma as separator, no extra spaces\\n"
                            "- Start directly with the header row\\n"
                            "Example format:\\n"
                            "polling_station,All India Anna Dravida Munnetra Kazhagam,Dravida Munnetra Kazhagam,Naam Tamilar Katchi,Makkal Needhi Maiam,Bharatiya Janata Party,Independent,NOTA,total\\n"
                            "14M,395,431,45,23,12,8,3,921\\n"
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
    }

def call_llm_and_get_csv(image_path):
    image_base64 = encode_image_base64(image_path)
    payload = build_prompt(image_base64)
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        res_json = response.json()
        if "choices" in res_json and len(res_json["choices"]) > 0:
            csv_text = res_json["choices"][0]["message"]["content"].strip()
            lines = csv_text.split('\n')
            csv_lines = [line.strip() for line in lines if line and ',' in line]
            cleaned_csv = '\n'.join(csv_lines)
            df = pd.read_csv(StringIO(cleaned_csv))
            df.columns = df.columns.str.strip()
            expected_columns = [
                "polling_station", "All India Anna Dravida Munnetra Kazhagam",
                "Dravida Munnetra Kazhagam", "Naam Tamilar Katchi",
                "Makkal Needhi Maiam", "Bharatiya Janata Party",
                "Independent", "NOTA", "total"
            ]
            if len(df.columns) == len(expected_columns):
                return df, None
            else:
                return None, "Unexpected columns count in CSV"
        else:
            return None, "No choices found in LLM response"
    else:
        return None, f"API Error: {response.text}"

def insert_ocr_data_into_db(df: pd.DataFrame):
    with Session(engine) as session:
        constituency = session.exec(select(Constituency).where(Constituency.name == DUMMY_CONSTITUENCY)).first()
        if not constituency:
            constituency = Constituency(name=DUMMY_CONSTITUENCY)
            session.add(constituency)
            session.commit()
            session.refresh(constituency)

        mandal = session.exec(select(Mandal).where(Mandal.name == DUMMY_MANDAL, Mandal.constituency_id == constituency.id)).first()
        if not mandal:
            mandal = Mandal(name=DUMMY_MANDAL, constituency_id=constituency.id)
            session.add(mandal)
            session.commit()
            session.refresh(mandal)

        village = session.exec(select(Village).where(Village.name == DUMMY_VILLAGE, Village.mandal_id == mandal.id)).first()
        if not village:
            village = Village(name=DUMMY_VILLAGE, mandal_id=mandal.id, constituency_id=constituency.id)
            session.add(village)
            session.commit()
            session.refresh(village)

        for idx, row in df.iterrows():
            polling_station_name = row["polling_station"]

            polling_station = session.exec(
                select(PollingStation).where(
                    PollingStation.name == polling_station_name,
                    PollingStation.village_id == village.id,
                    PollingStation.year == DUMMY_YEAR
                )
            ).first()
            if not polling_station:
                polling_station = PollingStation(
                    name=polling_station_name,
                    village_id=village.id,
                    year=DUMMY_YEAR
                )
                session.add(polling_station)
                session.commit()
                session.refresh(polling_station)

            for party_col in df.columns:
                if party_col in ("polling_station", "total", "NOTA"):
                    continue
                votes = row[party_col]
                if pd.isna(votes) or votes == 0:
                    continue
                party = session.exec(select(Party).where(Party.name == party_col)).first()
                if not party:
                    party = Party(name=party_col)
                    session.add(party)
                    session.commit()
                    session.refresh(party)

                candidate_name = f"{party_col} Candidate"
                candidate = session.exec(
                    select(Candidate).where(
                        Candidate.name == candidate_name,
                        Candidate.party_id == party.id,
                        Candidate.constituency_id == constituency.id
                    )
                ).first()
                if not candidate:
                    candidate = Candidate(
                        name=candidate_name,
                        party_id=party.id,
                        constituency_id=constituency.id
                    )
                    session.add(candidate)
                    session.commit()
                    session.refresh(candidate)

                vote_count = VoteCount(
                    polling_station_id=polling_station.id,
                    candidate_id=candidate.id,
                    votes=int(votes)
                )
                session.add(vote_count)

        session.commit()

def get_votes(
    location_type: str,
    location_id: int,
    candidate_id: Optional[int] = None,
    party_id: Optional[int] = None,
    constituency_id: Optional[int] = None
) -> int:
    with Session(engine) as session:
        query = select(VoteCount).join(PollingStation).join(Candidate)

        if location_type == "mandal":
            query = query.join(Village, PollingStation.village_id == Village.id)\
                         .join(Mandal, Village.mandal_id == Mandal.id)\
                         .where(Mandal.id == location_id)
        elif location_type == "village":
            query = query.join(Village, PollingStation.village_id == Village.id)\
                         .where(Village.id == location_id)
        elif location_type == "booth":
            query = query.where(PollingStation.id == location_id)
        else:
            raise HTTPException(status_code=400, detail="Invalid location_type. Choose from 'mandal', 'village', 'booth'.")

        if candidate_id:
            query = query.where(VoteCount.candidate_id == candidate_id)

        if party_id:
            query = query.where(Candidate.party_id == party_id)

        if constituency_id:
            query = query.where(Candidate.constituency_id == constituency_id)

        vote_counts = session.exec(query).all()
        total_votes = sum(v.votes for v in vote_counts)
        return total_votes


# --- FastAPI Routes ---

@app.post("/upload")
async def upload_pdf(pdf: UploadFile = File(...)):
    # Save PDF
    pdf_path = os.path.join(IMAGE_DIR, pdf.filename)
    with open(pdf_path, "wb") as f:
        content = await pdf.read()
        f.write(content)

    # Convert PDF to images
    image_paths = convert_pdf_to_images(pdf_path)

    results = []
    for img_path in image_paths:
        df, error = call_llm_and_get_csv(img_path)
        if df is not None:
            insert_ocr_data_into_db(df)
            results.append({"image": os.path.basename(img_path), "rows": len(df), "status": "Inserted"})
        else:
            results.append({"image": os.path.basename(img_path), "status": "Failed", "error": error})

    return {"message": "Processing completed", "results": results}


@app.get("/get_votes")
def api_get_votes(
    location_type: str = Query(..., description="Location type: mandal, village, booth"),
    location_id: int = Query(..., description="ID of the location"),
    candidate_id: Optional[int] = Query(None, description="Candidate ID"),
    party_id: Optional[int] = Query(None, description="Party ID"),
    constituency_id: Optional[int] = Query(None, description="Constituency ID"),
):
    total_votes = get_votes(location_type, location_id, candidate_id, party_id, constituency_id)
    return {"total_votes": total_votes}


@app.get("/results")
def get_results():
    with Session(engine) as session:
        vote_counts = session.exec(select(VoteCount)).all()
        results = []
        for v in vote_counts[:20]:  # limit to 20 results for preview
            results.append({
                "id": v.id,
                "polling_station_id": v.polling_station_id,
                "candidate_id": v.candidate_id,
                "votes": v.votes
            })
    return {"total_vote_counts": len(vote_counts), "sample_results": results}

@app.get("/")
def root():
    return {"message": "Election Results PDF Processor API", "endpoints": ["/upload", "/get_votes", "/results"]}
@app.get("/table/{table_name}")
def get_table_data(table_name: str):
    # Map valid table names to their SQLModel classes
    valid_tables = {
        "constituency": Constituency,
        "mandal": Mandal,
        "village": Village,
        "pollingstation": PollingStation,
        "party": Party,
        "candidate": Candidate,
        "votecount": VoteCount,
    }

    table_name_lower = table_name.lower()
    if table_name_lower not in valid_tables:
        raise HTTPException(status_code=400, detail="Invalid table name")

    model = valid_tables[table_name_lower]
    with Session(engine) as session:
        results = session.exec(select(model)).all()
        # Convert results (SQLModel objects) to list of dicts
        return {
            "table": table_name_lower,
            "row_count": len(results),
            "rows": [r.dict() for r in results]
        }