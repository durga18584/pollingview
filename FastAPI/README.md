# Election Results PDF Processor API

A FastAPI application that extracts election results from PDF documents using AI vision models and stores them in a SQLite database for querying.

## Features

- **PDF Upload & Processing**: Upload PDF files and automatically extract election data
- **AI-Powered OCR**: Uses OpenRouter's vision models to extract structured data from PDF images
- **Database Storage**: Stores results in SQLite database with proper relational structure
- **Vote Querying**: Query votes by location type (mandal, village, booth) and filters
- **Table Data Access**: View data from all database tables
- **CORS Enabled**: Ready for frontend integration

## Setup

### Prerequisites

- Python 3.7+
- OpenRouter API key (free tier available)

### Installation

1. **Clone or copy the code into a folder:**
   ```bash
   cd C:\Users\YourName\FastAPI
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate.bat
   ```

3. **Install dependencies:**
   ```bash
   pip install fastapi uvicorn sqlmodel pymupdf pandas requests
   ```

4. **Set your OpenRouter API key:**
   ```bash
   set OPENROUTER_API_KEY=your_api_key_here
   ```
   
   **Note:** The app currently has a hardcoded API key. For production, replace the `OPENROUTER_API_KEY` variable in `app.py` with your own key.

5. **Run the API server:**
   ```bash
   uvicorn app:app --reload
   ```

   You should see:
   ```
   INFO:     Uvicorn running on http://127.0.0.1:8000
   ```

## API Endpoints

### 1. Root Endpoint
- **GET** `http://127.0.0.1:8000/`
- Returns API information and available endpoints

### 2. Upload PDF
- **POST** `http://127.0.0.1:8000/upload`
- **Body:** Form data with PDF file
- **Description:** Uploads a PDF, converts to images, extracts election data using AI, and stores in database
- **Response:** Processing results for each page

**Example using curl:**
```bash
curl -X POST "http://127.0.0.1:8000/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "pdf=@your_election_results.pdf"
```

### 3. Query Votes
- **GET** `http://127.0.0.1:8000/get_votes`
- **Parameters:**
  - `location_type` (required): "mandal", "village", or "booth"
  - `location_id` (required): ID of the location
  - `candidate_id` (optional): Filter by specific candidate
  - `party_id` (optional): Filter by specific party
  - `constituency_id` (optional): Filter by specific constituency
- **Description:** Get total votes for a specific location with optional filters

**Examples:**
```bash
# Get all votes for mandal ID 1
GET http://127.0.0.1:8000/get_votes?location_type=mandal&location_id=1

# Get votes for specific candidate in village ID 2
GET http://127.0.0.1:8000/get_votes?location_type=village&location_id=2&candidate_id=5

# Get votes for specific party in booth ID 3
GET http://127.0.0.1:8000/get_votes?location_type=booth&location_id=3&party_id=2
```

### 4. View Results
- **GET** `http://127.0.0.1:8000/results`
- **Description:** Get sample vote count data (limited to 20 results for preview)

### 5. Table Data Access
- **GET** `http://127.0.0.1:8000/table/{table_name}`
- **Available tables:** constituency, mandal, village, pollingstation, party, candidate, votecount
- **Description:** View all data from a specific database table

**Examples:**
```bash
# View all candidates
GET http://127.0.0.1:8000/table/candidate

# View all parties
GET http://127.0.0.1:8000/table/party

# View all vote counts
GET http://127.0.0.1:8000/table/votecount
```

## Database Schema

The application uses the following database structure:

- **Constituency**: Electoral constituencies
- **Mandal**: Administrative divisions within constituencies
- **Village**: Villages within mandals
- **PollingStation**: Individual polling stations
- **Party**: Political parties
- **Candidate**: Candidates with party and constituency associations
- **VoteCount**: Vote counts for candidates at polling stations

## Usage Workflow

1. **Start the server** using `uvicorn app:app --reload`
2. **Upload a PDF** containing election results using the `/upload` endpoint
3. **Query the data** using `/get_votes` with appropriate location types and IDs
4. **View table data** using `/table/{table_name}` to explore the stored information

## Configuration

The application uses the following default settings (configurable in `app.py`):

- **Vision Model**: `mistralai/mistral-small-3.2-24b-instruct:free`
- **Database**: `results.db` (SQLite)
- **Image Directory**: `pdf_images/`
- **Dummy Data**: Sample constituency, mandal, village, and year (2024)

## Error Handling

The API includes proper error handling for:
- Invalid location types
- Missing required parameters
- File upload issues
- Database connection problems
- AI model processing errors

## CORS Support

The API is configured with CORS middleware to allow cross-origin requests, making it suitable for frontend integration.

