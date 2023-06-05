// 画布
const canvas = document.getElementById('canvas')
const ctx = canvas.getContext('2d')
// 图片
const img = document.getElementById('image')
// 按钮
const button = document.getElementById('button')
// 预测结果
const prediction = document.getElementById('prediction')
// 下拉列表
const select = document.getElementById('model')
// 置信度
const probability = document.getElementById('probability')

// 给画板填充颜色（黑色）
ctx.fillStyle = '000000'
ctx.fillRect(0, 0, canvas.width, canvas.height)

// 设置画笔颜色（白色），线宽（5）
ctx.strokeStyle = '#ffffff'
ctx.lineWidth = 5

// 按下标记
let isDrawing = false

// 设置画笔颜色（白色）
const lineColor = '#ffffff'

// 设置画笔线宽
const lineWidth = 5

function getMousePos(canvas, event) {
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const offsetX = (event.clientX - rect.left) * scaleX
    const offsetY = (event.clientY - rect.top) * scaleY
    return {offsetX, offsetY}
}

// 鼠标按下
function startDrawing(event) {
    isDrawing = true
    const {offsetX, offsetY} = getMousePos(canvas, event)
    ctx.beginPath()
    ctx.moveTo(offsetX, offsetY)
}

canvas.addEventListener('mousedown', startDrawing, true)

// 鼠标移动
function draw(event) {
    if (!isDrawing) {
        return
    }

    const {offsetX, offsetY} = getMousePos(canvas, event)
    ctx.lineTo(offsetX, offsetY)
    ctx.strokeStyle = lineColor
    ctx.lineWidth = lineWidth
    ctx.stroke()
}

canvas.addEventListener('mousemove', draw, false)

// 鼠标松开
function stopDrawing() {
    isDrawing = false
}

canvas.addEventListener('mouseup', stopDrawing, false)
canvas.addEventListener('mouseout', stopDrawing, false)

// 按钮点击事件
function click() {
    // 将画布转换为图片，canvas.toDataURL 返回的是一串Base64编码的URL
    let url = canvas.toDataURL('image/jpg')
    console.log("Pic url : " + url)

    console.log("model : " + select.value)

    let request_url = `http://127.0.0.1:5000/predict/${select.value}`
    console.log("request url : " + request_url)

    // 以json格式向后端发送图片的url
    fetch(request_url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            url: url
        })
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok')
            }

            return response.json()
        })
        .then(data => {
            if (data.flag === 'success') {
                // console.log(data.img);
                const imgBase64 = data.img;
                img.src = 'data:image/png;base64,' + imgBase64;

                // 显示预测结果
                prediction.innerHTML = `预测结果：${data.results['prediction(s)']}`
                // 显示置信度
                probability.innerHTML = `置信度：${data.results['probability']}`
            } else if (data.flag === 'fail') {
                console.log('fail')
                console.log(data.results.info)

                alert('fail to predict. Go to console for more information.')
            }
        })
        .catch(error => {
            console.error('Error:', error)
        })
        .finally(() => {
            // 隐藏进度条
            // progress.style.display = 'none';
        })

    // 清空画布
    canvas.width = canvas.width
    canvas.height = canvas.height

    // 给画板填充颜色，为下一个用户写下一个字作准备
    ctx.fillStyle = '000000'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
}

button.addEventListener('click', click)
