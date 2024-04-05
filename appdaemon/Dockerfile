#install python en pip
RUN apt-get install python3 -y\
    python3-pip -y\
    python3-venv -y

WORKDIR /root
ENV VIRTUAL_ENV=/root/venv/day_ahead
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt
