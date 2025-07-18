# Home Assistant provides the correct Alpine base for every CPU arch
ARG BUILD_FROM
FROM $BUILD_FROM

# Install Python and pip
RUN apk add --no-cache python3 py3-pip

# Copy runtime files
COPY run.sh /run.sh
COPY predai_main.py /opt/predai/predai_main.py
COPY requirements.txt /tmp/requirements.txt

RUN chmod a+x /run.sh \
 && pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /opt/predai
CMD [ "/run.sh" ]
