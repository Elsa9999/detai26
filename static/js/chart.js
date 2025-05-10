// static/js/chart.js
// Chart.js config: all three metrics as bars, dual axes, vibrant colors

// Khởi tạo biểu đồ benchmark
function initBenchmarkChart() {
    const canvas = document.getElementById('benchmarkChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const methods = JSON.parse(canvas.dataset.methods || '[]');
    const insertTimes = JSON.parse(canvas.dataset.insertTimes || '[]');
    const searchTimes = JSON.parse(canvas.dataset.searchTimes || '[]');
    const memories = JSON.parse(canvas.dataset.memories || '[]');

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: methods,
            datasets: [
                {
                    label: 'Insert Time (ms)',
                    data: insertTimes,
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Search Time (ms)',
                    data: searchTimes,
                    backgroundColor: 'rgba(255, 99, 132, 0.5)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Memory Usage (MB)',
                    data: memories,
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Khởi tạo biểu đồ khi trang load xong
document.addEventListener('DOMContentLoaded', function() {
    initBenchmarkChart();
});
