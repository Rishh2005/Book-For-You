import uvicorn

from cab_booking_ai.api import app

if __name__ == '__main__':
    uvicorn.run('cab_booking_ai.api:app', host='0.0.0.0', port=8000, reload=True)
