const app = require('express')()
let port = process.env.PORT || 3000;
const importData = require("./data.json")
const tfn = require("@tensorflow/tfjs-node")
const tf = require("@tensorflow/tfjs")
const handler = tfn.io.fileSystem("./model/model.json");
const bodyParser = require("body-parser")


app.use(bodyParser.json())
app.use(bodyParser.urlencoded({ extended: false }))


const model = async(temp, ph) => {
    var data
    const model = await tf.loadLayersModel(handler).then(m => {
        var results = m.predict(tf.tensor2d([temp, ph], [1, 2]))
        data = results.dataSync()[0]
        // console.log(data)
        return data
    })
    return data
}

app.get("/:temp/:ph", async (req, res) => {
    var temp = req.params.temp
    var ph = req.params.ph
    var ans = await model(parseFloat(temp), parseFloat(ph))
    res.json({ ans })
})


app.get("/players", (req, res) => {
    res.send(importData)
})

app.listen(port, () => {
    console.log("Server running onn port: ", port)
})