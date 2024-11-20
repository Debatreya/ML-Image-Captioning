const express = require('express');
const app = express();

app.use('/',(req,res)=>{
    res.send('ML Project Backend')
});

const port = 5000;

app.use('/runModel',(req,res)=>{
    
})

app.listen(port,()=>{
    console.log(`Server listening on port : ${port}`);  
})