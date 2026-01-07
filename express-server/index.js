const express = require('express');
const Busboy = require('busboy');
const axios = require('axios');
const path = require('path');

const app = express();
const PORT = 3000;

const WEBHDFS_HOST =  'namenode';
const WEBHDFS_PORT = '9870';
const WEBHDFS_USER = 'root';
const HDFS_BASE_PATH = '/'; 

app.post('/upload', (req, res) => {
    const busboy = Busboy({ headers: req.headers });

    busboy.on('file', async (fieldname, file, info) => {
        const { filename, encoding, mimeType } = info;
        console.log(`File [${fieldname}]: filename: ${filename}, encoding: ${encoding}, mimeType: ${mimeType}`);
        file.pause();

        try {
            const hdfsPath = HDFS_BASE_PATH.endsWith('/') ? `${HDFS_BASE_PATH}${filename}` : `${HDFS_BASE_PATH}/${filename}`;
            const createUrl = `http://${WEBHDFS_HOST}:${WEBHDFS_PORT}/webhdfs/v1${hdfsPath}?op=CREATE&overwrite=true&user.name=${WEBHDFS_USER}&noredirect=true`;
            
            console.log(`Initiating HDFS upload to: ${createUrl}`);
            
            const response = await axios.put(createUrl);
            
            if (response.data && response.data.Location) {
                const dataNodeUrl = response.data.Location;
                console.log(`Redirected to DataNode: ${dataNodeUrl}`);
                file.resume();
                
                await axios.put(dataNodeUrl, file, {
                    headers: {
                        'Content-Type': 'application/octet-stream',
                        'Transfer-Encoding': 'chunked' 
                    },
                    maxBodyLength: Infinity,
                    maxContentLength: Infinity
                });

                console.log(`Successfully uploaded ${filename} to HDFS`);
            } else {
                console.error('Failed to get DataNode redirection URL');
                file.resume();
            }

        } catch (error) {
            console.error('Error uploading to HDFS:', error.message);
            if (error.response) {
                console.error('HDFS Response:', error.response.data);
            }
            if (file.readable) {
                file.resume();
            }
        }
    });

    busboy.on('finish', () => {
        res.status(200).send('Upload complete');
    });

    busboy.on('error', (err) => {
        console.error('Busboy error:', err);
        res.status(500).send('Upload failed');
    });

    req.pipe(busboy);
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
