window.onload=function(){
	const canvas = document.getElementById('myCanvas');//1.获取画布
	const ctx = canvas.getContext('2d');//2.获取上下文
	// const strokeStyleSelect = document.getElementById('strokeColorSelect');//改变颜色控件
	// const strokeLineWidth = document.getElementById('strokeLineWidth');//改变线条宽度控件
	const img = document.getElementById("image");
	const btn = document.getElementById("btn")
	const prediction = document.getElementById('prediction');
	
	// 给画板填充颜色
	ctx.fillStyle = '000000';
	ctx.fillRect(0, 0, canvas.width, canvas.height)
	
	//按下标记
	let isOnOff = false;
	let oldX = null;
	let oldY = null;
	
	//设置画笔颜色
	let lineColor = '#ffffff';  //默认线条颜色为黑色
	
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
	
	
	function send() {
	    // 新Image对象，可以理解为DOM  
	    // var image = new Image(); 
	    // canvas.toDataURL 返回的是一串Base64编码的URL
	    // 指定格式 JPG  
	    img.src = canvas.toDataURL("image/jpg");  
	    // return image;

		console.log(img.src)

		// 以json格式向后端发送图片的url
		fetch(`http://127.0.0.1:5000/test`, {
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
				let predict = null;
				if (data.result === 'success') {
					prediction.innerHTML = '您书写的数字是：' + data.predict;
				} else if (data.result === 'fail') {
					console.log('fail')
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
