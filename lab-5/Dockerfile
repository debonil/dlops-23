FROM python:3.10
COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
CMD python M21AIE225_Lab_Assignment_5.py