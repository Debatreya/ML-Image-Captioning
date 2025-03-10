require('express-async-errors');
const cors = require('cors')
const express = require('express');
const app = express();

const multer = require('multer');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const uploadFolderPath = path.join(__dirname, '..', 'images');
const moragn = require('morgan');

const upload = multer({
    dest: uploadFolderPath,
});

app.use(moragn('tiny'));

app.use(cors());

app.post('/runModel', upload.single('image'), (req, res) => {

    // Check if a file was uploaded
    if (!req.file) {
        return res.status(400).send('No file uploaded.');
    }

    const imagesFolder = path.join(__dirname, '..', 'images');
    const outputFolder = path.join(__dirname, '..', 'output');

    
    fs.readdir(outputFolder, (err, files) => {
        if (err) return res.status(500).send('Failed to read output folder.');

        files.forEach(file => {
            const filePath = path.join(outputFolder, file);
            try {
                fs.unlinkSync(filePath);
                console.log(`Deleted file: ${file}`);
            } catch (err) {
                console.error(`Failed to delete ${file}:`, err);
            }
        });
    });
    
    // The temp file uploaded by multer
    const tempFilePath = req.file.path;
    // The target path with the original filename
    const targetFilePath = path.join(imagesFolder, req.file.originalname);

    // Move the file from the temp folder to the target location
    fs.rename(tempFilePath, targetFilePath, (err) => {
        if (err) {
            console.error('Error moving file:', err);
            return res.status(500).send('Failed to move uploaded file.');
        }

        console.log('File moved successfully.');

        // Clean the "images" folder (delete existing files, except the newly uploaded file)
        fs.readdir(imagesFolder, (err, files) => {
            if (err) return res.status(500).send('Failed to read images folder.');

            files.forEach(file => {
                // Skip the newly uploaded file
                if (file !== req.file.originalname) {
                    const filePath = path.join(imagesFolder, file);
                    try {
                        fs.unlinkSync(filePath);  // Delete files except the uploaded one
                        console.log(`Deleted file: ${file}`);
                    } catch (err) {
                        console.error(`Failed to delete ${file}:`, err);
                    }
                }
            });

            // Now that the file is uploaded and moved, run the Python script
            const pythonScriptPath = path.join(__dirname, '..', 'run.py');
            const command = `python ${pythonScriptPath}`;

            // Run the Python script
            exec(command, (error, stdout, stderr) => {
                if (error) {
                    console.error('Error running the script:', stderr);
                    return res.status(500).send('Error running the model script.');
                }

                // console.log('Python script output:', stdout);

                // After running the model, check for the output file
                fs.readdir(outputFolder, (err, outputFiles) => {
                    if (err || outputFiles.length === 0) {
                        return res.status(500).send('No output image found.');
                    }

                    const outputImage = path.join(outputFolder, outputFiles[0]);

                    // Send the output image to the client
                    res.sendFile(outputImage, err => {
                        if (err) {
                            console.error('Error sending the output image:', err);
                        }
                    });
                });
            });
        });
    });
});


// /health endpoint
app.get('/health', (req, res) => {
    res.send('Server is running.');
});



// Start the server
app.listen(5000, () => {
    console.log('Server is running on http://localhost:5000');
});
