const express = require('express');
const Busboy = require('busboy');
const axios = require('axios');
const unzipper = require('unzipper');
const { PassThrough } = require('stream');
const path = require('path');

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 3001;
const WH_HOST = process.env.WEBHDFS_HOST || 'namenode';
const WH_PORT = process.env.WEBHDFS_PORT || '9870';
const WH_USER = process.env.WEBHDFS_USER || 'root';
const WH_DN_PORT = process.env.WEBHDFS_DATANODE_PORT || null;

async function hdfsDataNodeUrl(hdfsFilePath) {
    const url = `http://${WH_HOST}:${WH_PORT}/webhdfs/v1${hdfsFilePath}?op=CREATE&overwrite=true&user.name=${WH_USER}&noredirect=true`;
    const res = await axios.put(url);
    if (!res.data?.Location) throw new Error('No DataNode redirect from NameNode');
    let location = res.data.Location;
    if (WH_DN_PORT) {
        const u = new URL(location);
        u.hostname = WH_HOST;
        u.port = WH_DN_PORT;
        location = u.toString();
    }
    return location;
}

async function putToHdfs(stream, hdfsFilePath) {
    const dataNodeUrl = await hdfsDataNodeUrl(hdfsFilePath);
    await axios.put(dataNodeUrl, stream, {
        headers: {
            'Content-Type': 'application/octet-stream',
            'Transfer-Encoding': 'chunked',
        },
        maxBodyLength: Infinity,
        maxContentLength: Infinity,
    });
    console.log(`Uploaded → HDFS ${hdfsFilePath}`);
}

// POST /upload — multipart file upload straight to HDFS
app.post('/upload', (req, res) => {
    const busboy = Busboy({ headers: req.headers });

    busboy.on('file', async (fieldname, file, info) => {
        const hdfsPath = `/logs/raw/${info.filename}`;
        file.pause();
        try {
            const dataNodeUrl = await hdfsDataNodeUrl(hdfsPath);
            file.resume();
            await axios.put(dataNodeUrl, file, {
                headers: {
                    'Content-Type': 'application/octet-stream',
                    'Transfer-Encoding': 'chunked',
                },
                maxBodyLength: Infinity,
                maxContentLength: Infinity,
            });
            console.log(`Uploaded ${info.filename} → HDFS ${hdfsPath}`);
        } catch (err) {
            console.error('Upload error:', err.message);
            file.resume();
        }
    });

    busboy.on('finish', () => res.status(200).send('Upload complete'));
    busboy.on('error', () => res.status(500).send('Upload failed'));
    req.pipe(busboy);
});

// POST /ingest/kaggle — stream a Kaggle dataset zip directly to HDFS (no local disk)
//
// Body (JSON, all fields optional):
//   owner   – Kaggle dataset owner  (default: "eliasdabbas")
//   dataset – Kaggle dataset slug   (default: "web-server-access-logs")
//   hdfsDir – HDFS destination dir  (default: "/logs/raw")
//
// Required env vars: KAGGLE_USERNAME, KAGGLE_KEY
//
// Example:
//   curl -X POST http://localhost:3000/ingest/kaggle \
//        -H 'Content-Type: application/json' \
//        -d '{"owner":"eliasdabbas","dataset":"web-server-access-logs","hdfsDir":"/logs/raw"}'
app.post('/ingest/kaggle', async (req, res) => {
    const KAGGLE_USERNAME = "sarveshwarb";
    const KAGGLE_KEY = "KGAT_42eda578c9b53dea20f4a146d9c5a9be";

    if (!KAGGLE_USERNAME || !KAGGLE_KEY) {
        return res.status(500).json({ error: 'KAGGLE_USERNAME and KAGGLE_KEY env vars are required' });
    }

    const owner = req.body?.owner || 'eliasdabbas';
    const dataset = req.body?.dataset || 'web-server-access-logs';
    const hdfsDir = req.body?.hdfsDir || '/logs/raw';

    const kaggleUrl = `https://www.kaggle.com/api/v1/datasets/download/${owner}/${dataset}`;
    console.log(`Fetching ${kaggleUrl} → HDFS ${hdfsDir}`);

    try {
        const kaggleRes = await axios.get(kaggleUrl, {
            auth: { username: KAGGLE_USERNAME, password: KAGGLE_KEY },
            responseType: 'stream',
            maxRedirects: 5,
        });

        const uploaded = [];
        const entryPromises = [];

        await new Promise((resolve, reject) => {
            kaggleRes.data
                .pipe(unzipper.Parse())
                .on('entry', (entry) => {
                    if (entry.type !== 'File') {
                        entry.autodrain();
                        return;
                    }

                    const hdfsPath = `${hdfsDir}/${path.basename(entry.path)}`;
                    console.log(`Streaming ${entry.path} → HDFS ${hdfsPath}`);

                    // Pipe entry into a PassThrough immediately so the zip parser
                    // is never stalled by backpressure from our async HDFS call.
                    const pass = new PassThrough();
                    entry.pipe(pass);

                    entryPromises.push(
                        putToHdfs(pass, hdfsPath)
                            .then(() => uploaded.push(hdfsPath))
                            .catch(reject)
                    );
                })
                .on('finish', () => {
                    Promise.all(entryPromises).then(() => resolve()).catch(reject);
                })
                .on('error', reject);
        });

        res.status(200).json({ message: 'Ingest complete', files: uploaded });
    } catch (err) {
        console.error('Kaggle ingest error:', err.message);
        if (err.response) console.error('HTTP', err.response.status, err.response.statusText);
        res.status(500).json({ error: err.message });
    }
});

app.listen(PORT, () => console.log(`Server listening on port ${PORT}`));
