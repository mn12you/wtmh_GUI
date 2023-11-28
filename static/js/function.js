var current_check = 0;

// 解析数据
var predict_list = JSON.parse(document.getElementById("pred_list").dataset.pred_list);
var cwt_list = JSON.parse(document.getElementById("cwt_list").dataset.cwt_list);
var seg_list = JSON.parse(document.getElementById("seg_list").dataset.seg_list);
console.log(predict_list)
console.log(cwt_list)
console.log(seg_list)

// Initialize ECG Chart
var ecgCtx = document.getElementById('ecgChart').getContext('2d');
var ecgChart = new Chart(ecgCtx, {
    type: 'line',
    data: {
        datasets: [{
            label: 'ECG Data',
            data: [],
            borderColor: 'rgba(0, 123, 255, 0.6)',
            fill: false,
        }]
    },
    options: {
        scales: {
            x: {
                type: 'linear',
                position: 'bottom'
            }
        }
    }
});

// Initialize Heatmap Chart
var heatmapCtx = document.getElementById('heatmap').getContext('2d');
var heatmapChart;

function generateHeatmapData(cwtData) {
    var data = [];
    for (let i = 0; i < 112; i++) {
        for (let j = 0; j < 112; j++) {
            var value = cwtData[i][j];
            data.push({ x: j, y: i, v: value*1.1 });
        }
    }
    return data;
}
var label = document.getElementById('predictionlabel');
function updatelabel(newData) {
    label.textContent = "Prediction Label: " + newData;
    
}

function updateChart(data) {
    ecgChart.data.datasets[0].data = data;
    ecgChart.update();
}

function updateHeatmap(data) {
    heatmapChart.data.datasets[0].data = data;
    heatmapChart.update();
}

function updateVisualizations() {
    updateChart(seg_list[current_check].map((y, x) => ({ x, y })));
    updateHeatmap(generateHeatmapData(cwt_list[current_check]));
    updatelabel(predict_list[current_check])
}

document.addEventListener('DOMContentLoaded', function () {
    heatmapChart = new Chart(document.getElementById('heatmap').getContext('2d'), {
        type: 'matrix',
        data: {
            datasets: [{
                label: 'Grayscale Heatmap',
                data: generateHeatmapData(cwt_list[current_check]),
                backgroundColor: function(context) {
                    var value = context.dataset.data[context.dataIndex].v;
                    var alpha = value ; // 根据您的数据范围调整
                    return `rgba(0, 0, 0, ${alpha})`;
                },
                width: function(context) {
                    return context.chart.width / 112;
                },
                height: function(context) {
                    return context.chart.height / 112;
                }
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { type: 'linear', display: false },
                y: { type: 'linear', display: false }
            },
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
            }
        }
    });

    // 初始化可视化
    updateVisualizations();
});


// 获取按钮元素并绑定事件
const previous_butt = document.getElementById('upindexbtn');
const next_butt = document.getElementById('downindexbtn');
previous_butt.addEventListener('click', function() {
    if (current_check > 0) {
        current_check--;
        updateVisualizations();
    }
});
next_butt.addEventListener('click', function() {
    if (current_check < seg_list.length - 1) {
        current_check++;
        updateVisualizations();
    }
});
