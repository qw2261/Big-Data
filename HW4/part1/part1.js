var data = [80, 100, 56, 120, 180, 30, 40, 120, 160];
var svgWidth = 500, svgHeight = 300;
// The required padding between bars is 5px.
// The label must locate 2px above the middle of each bar.

var svg = d3.select('svg')
        .attr("width", svgWidth)
        .attr("height", svgHeight);

var barWidth = 50;

var barChart = svg.selectAll("rect")
        .data(data)
        .enter()
        .append("rect")
        .attr("class", "bar")
        // .attr('y', function(d){return svgHeight - d})
        // .attr('x', function(d,i){return i*(svgWidth / 9)})
        .attr('transform', function(d, i){return "translate(" + (i * (barWidth + 5)) + "," + (svgHeight - d) + ")"})
        .attr('width', barWidth)
        .attr('height', function(d){return d})
        .attr("fill", "#CC6450");

var g = svg.selectAll("text")
        .data(data)
        .enter()
        .append("text")
        .attr('y', function(d){return svgHeight - d - 2})
        .attr('x', function(d,i){return i * (barWidth + 5) + barWidth / 5})
        .text(function(d){return d});