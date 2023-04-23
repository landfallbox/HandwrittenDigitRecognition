window.onload=function(){
	// 画布
	const canvas = document.getElementById('myCanvas');
	// 上下文
	const ctx = canvas.getContext('2d');
	// 图片
	const img = document.getElementById("image");
	// 按钮
	const btn = document.getElementById("btn")
	// 预测结果
	const prediction = document.getElementById('prediction');
	
	// 给画板填充颜色（黑色）
	ctx.fillStyle = '000000';
	ctx.fillRect(0, 0, canvas.width, canvas.height)
	
	//按下标记
	let isOnOff = false;
	let oldX = null;
	let oldY = null;
	
	//设置画笔颜色（白色）
	let lineColor = '#ffffff';
	
	//设置画笔线宽
	let lineWidth = 5;
	
	//添加鼠标移动事件
	canvas.addEventListener('mousemove', draw, false);
	//添加鼠标按下事件
	canvas.addEventListener('mousedown', down, true);
	//添加鼠标松开事件
	canvas.addEventListener('mouseup', up, false);
	
	// 监听鼠标点击
	btn.addEventListener("click", send);
	
	function down(event) {
		isOnOff = true;
		oldX = event.clientX;
		oldY = event.clientY;
	}
	
	function up() {
		isOnOff = false;
	}
	
	function draw(event) {
		if (isOnOff === true) {
			let newX = event.clientX;
			let newY = event.clientY;
	
			ctx.beginPath();
			ctx.moveTo(oldX, oldY);
			ctx.lineTo(newX, newY);
			ctx.strokeStyle = lineColor;
			ctx.lineWidth = lineWidth;
			ctx.lineCap = 'round';
			ctx.stroke();
	
			oldX = newX;
			oldY = newY;
		}
	}
	
	// 按钮点击事件
	function send() {
		// 将画布转换为图片，canvas.toDataURL 返回的是一串Base64编码的URL，格式为 JPG
	    img.src = canvas.toDataURL("image/jpg");

		console.log(img.src)

		// 以json格式向后端发送图片的url
		fetch(`http://127.0.0.1:5000/predict_VGGNet`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({
				url: img.src
			})
		})
			.then(response => {
				if (!response.ok) {
					throw new Error('Network response was not ok');
				}
				return response.json();
			})
			.then(data => {
				if (data.flag === 'success') {
					prediction.innerHTML = '预测结果：' + data.results['prediction(s)'];
				} else if (data.flag === 'fail') {
					console.log('fail')
					console.log(data.results.info)
					alert('fail to predict. Go to console for more information.');
				}
			})
			.catch(error => {
				console.error('Error:', error);
			});
		
		// 清空画布
		// canvas.clearRect(0, 0, canvas.width, canvas.height);
		canvas.width = canvas["width"];
		canvas.height = canvas["height"];
		
		// 给画板填充颜色，为下一个用户写下一个字作准备
		ctx.fillStyle = '000000';
		ctx.fillRect(0, 0, canvas.width, canvas.height)
	}
};
