FROM python:3.7-alpine
WORKDIR /app
RUN apk --upgrade add make automake gcc g++ subversion python3-dev
RUN pip install numpy pandas nltk sklearn
COPY Recommendataion(final.py) .