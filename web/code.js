var svg = d3.select("svg"),
  width = +svg.node().getBoundingClientRect().width,
  height = +svg.node().getBoundingClientRect().height;

// svg objects
var link, node, circles;
// the data - an object with nodes and links
var graph;

// sets the color of the nodes
var color = d3.scaleOrdinal(d3.schemeCategory20);

const STATIC_SIZE = 8;
const STROKE_COLOR = "#aaa";

var degreeSize;

const searchParams = new URLSearchParams(location.search);
if (searchParams.has("hideToolbar")) {
  document.querySelector(".controls").classList.add("hide");
  document.querySelector("header").classList.add("hide");
}

// load the data
d3.json("./results/log/2023y10m24d/graph_original.json", function (error, _graph) {
  if (error) throw error;
  graph = _graph;

  // Linear scale for degree centrality.
  degreeSize = d3
    .scaleLinear()
    .domain([
      d3.min(graph.nodes, function (d) {
        return d.degree;
      }),
      d3.max(graph.nodes, function (d) {
        return d.degree;
      }),
    ])
    .range([8, 25]);

  initializeDisplay();
  initializeSimulation();
  initializeSliders();
});

//////////// FORCE SIMULATION ////////////

// force simulator
var simulation = d3.forceSimulation();

// set up the simulation and event to update locations after each tick
function initializeSimulation() {
  simulation.nodes(graph.nodes);
  initializeForces();
  simulation.on("tick", ticked);
}

function initializeSliders() {
  const thresholdSelector = document.getElementById("threshold");
  thresholdSelector.setAttribute(
    "min",
    d3.min(graph.links, function (d) {
      return d.weight;
    })
  );
  thresholdSelector.setAttribute(
    "max",
    d3.max(graph.links, function (d) {
      return d.weight;
    })
  );
}

// values for all forces
var forceProperties = {
  center: {
    x: 0.5,
    y: 0.5,
  },
  charge: {
    enabled: true,
    strength: -30,
    distanceMin: 1,
    distanceMax: 2000,
  },
  collide: {
    enabled: true,
    strength: 0.7,
    iterations: 1,
    radius: 5,
  },
  forceX: {
    enabled: false,
    strength: 0.1,
    x: 0.5,
  },
  forceY: {
    enabled: false,
    strength: 0.1,
    y: 0.5,
  },
  link: {
    enabled: true,
    distance: 150,
    iterations: 1,
  },
};

// add forces to the simulation
function initializeForces() {
  // add forces and associate each with a name
  simulation
    .force("link", d3.forceLink())
    .force("charge", d3.forceManyBody())
    .force("collide", d3.forceCollide())
    .force("center", d3.forceCenter())
    .force("forceX", d3.forceX())
    .force("forceY", d3.forceY());
  // apply properties to each of the forces
  updateForces();
}

// apply new force properties
function updateForces() {
  // get each force by name and update the properties
  simulation
    .force("center")
    .x(width * forceProperties.center.x)
    .y(height * forceProperties.center.y);
  simulation
    .force("charge")
    .strength(forceProperties.charge.strength * forceProperties.charge.enabled)
    .distanceMin(forceProperties.charge.distanceMin)
    .distanceMax(forceProperties.charge.distanceMax);
  simulation
    .force("collide")
    .strength(
      forceProperties.collide.strength * forceProperties.collide.enabled
    )
    .radius(STATIC_SIZE)
    .iterations(forceProperties.collide.iterations);
  simulation
    .force("forceX")
    .strength(forceProperties.forceX.strength * forceProperties.forceX.enabled)
    .x(width * forceProperties.forceX.x);
  simulation
    .force("forceY")
    .strength(forceProperties.forceY.strength * forceProperties.forceY.enabled)
    .y(height * forceProperties.forceY.y);
  simulation
    .force("link")
    .id(function (d) {
      return d.id;
    })
    .distance(forceProperties.link.distance)
    .iterations(forceProperties.link.iterations)
    .links(forceProperties.link.enabled ? graph.links : []);

  // updates ignored until this is run
  // restarts the simulation (important if simulation has already slowed down)
  simulation.alpha(1).restart();
}

//////////// DISPLAY ////////////

// generate the svg objects and force simulation
function initializeDisplay() {
  // set the data and properties of link lines
  link = svg
    .append("g")
    .attr("class", "links")
    .selectAll("line")
    .data(graph.links)
    .enter()
    .append("line")
    // Sets link color
    .attr("stroke", STROKE_COLOR);

  // set the data and properties of node circles
  node = svg
    .append("g")
    .attr("class", "nodes")
    .selectAll("g")
    .data(graph.nodes)
    .enter()
    .append("g");

  circles = node
    .append("circle")
    .attr("fill", function (d) {
      return color(d.group);
    })
    .attr("stroke", "black")
    .attr("stroke-width", 0.75)
    .attr("r", function () {
      return STATIC_SIZE;
    })
    .call(
      d3
        .drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended)
    );

  // node tooltip
  node.append("title").text(function (d) {
    return d.name;
  });

  node
    .append("text")
    .attr("x", function () {
      const circle = this.parentElement.querySelector("circle");
      return parseInt(circle.getAttribute("r")) + 5;
    })
    .attr("y", 3)
    .text(function (d) {
      return d.name;
    });

  // Zooming function translates the size of the svg container.
  function zoomed() {
    const transform = `translate(${d3.event.transform.x}, ${d3.event.transform.y}) scale(${d3.event.transform.k})`;
    const nodes = document.querySelector(".nodes");
    const links = document.querySelector(".links");
    nodes.setAttribute("transform", transform);
    links.setAttribute("transform", transform);
  }

  // Call zoom for svg container.
  svg.call(d3.zoom().on("zoom", zoomed));

  // visualize the graph
  updateDisplay();
}

// update the display based on the forces (but not positions)
function updateDisplay() {
  node.selectAll("text").attr("x", function () {
    const circle = this.parentElement.querySelector("circle");
    return parseInt(circle.getAttribute("r")) + 5;
  });

  link.attr("opacity", forceProperties.link.enabled ? 1 : 0);
}

// update the display positions after each simulation tick
function ticked() {
  link
    .attr("x1", function (d) {
      return d.source.x;
    })
    .attr("y1", function (d) {
      return d.source.y;
    })
    .attr("x2", function (d) {
      return d.target.x;
    })
    .attr("y2", function (d) {
      return d.target.y;
    });

  node.attr("transform", function (d) {
    return `translate(${d.x},${d.y})`;
  });
  d3.select("#alpha_value").style("flex-basis", simulation.alpha() * 100 + "%");
}

//////////// UI EVENTS ////////////

function dragstarted(d) {
  if (!d3.event.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}

function dragged(d) {
  d.fx = d3.event.x;
  d.fy = d3.event.y;
}

function dragended(d) {
  if (!d3.event.active) simulation.alphaTarget(0.0001);
  d.fx = null;
  d.fy = null;
}

// update size-related forces
d3.select(window).on("resize", function () {
  width = +svg.node().getBoundingClientRect().width;
  height = +svg.node().getBoundingClientRect().height;
  updateForces();
});

// convenience function to update everything (run after UI input)
function updateAll() {
  updateForces();
  updateDisplay();
}

// edge weight slider

function setEdgeWeight(value) {
  var threshold = parseFloat(value);
  d3.select("#link_ThresholdSliderOutput").text(threshold);

  // Find the links that are at or above the threshold.
  var newData = [];
  graph.links.forEach(function (d) {
    if (d.weight >= threshold) {
      newData.push(d);
    }
  });

  // Data join with only those new links.
  link = link.data(newData, function (d) {
    return d.source + ", " + d.target;
  });
  link.exit().remove();
  var linkEnter = link
    .enter()
    .append("line")
    .attr("class", "link")
    .attr("stroke", STROKE_COLOR);
  link = linkEnter.merge(link);

  node = node.data(graph.nodes);

  // Restart simulation with new link data.
  simulation.nodes(graph.nodes).on("tick", ticked).force("link").links(newData);

  simulation.alphaTarget(0.1).restart();
}

function setCentrality(centrality) {
  var centralitySize = d3
    .scaleLinear()
    .domain([
      d3.min(graph.nodes, function (d) {
        return d[centrality];
      }),
      d3.max(graph.nodes, function (d) {
        return d[centrality];
      }),
    ])
    .range([8, 25]);
  circles.attr("r", function (d) {
    if (centrality === "static") {
      return STATIC_SIZE;
    }
    return centralitySize(d[centrality]);
  });
  // Recalculate collision detection based on selected centrality.
  simulation.force(
    "collide",
    d3.forceCollide().radius(function (d) {
      return centralitySize(d[centrality]);
    })
  );
  simulation.alphaTarget(0.1).restart();
}

function toggleTitleVisiblity() {
  const visibility = document.getElementById("title-visibility").checked;
  const nodes = document.querySelector(".nodes");
  if (visibility) {
    nodes.classList.remove("hide");
    return;
  }
  nodes.classList.add("hide");
}

// we need to handle a user gesture to use 'open'
document.getElementById("open-in-tab").onclick = () => {
  // grab your svg element
  const svg = document.querySelector("svg");
  svg.style.font = "10px sans-serif";
  // convert to a valid XML source
  const as_text = new XMLSerializer().serializeToString(svg);
  // store in a Blob
  const blob = new Blob([as_text], { type: "image/svg+xml" });
  // create an URI pointing to that blob
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "graph.svg";

  // Click handler that releases the object URL after the element has been clicked
  // This is required for one-off downloads of the blob content
  const clickHandler = () => {
    setTimeout(() => {
      URL.revokeObjectURL(url);
      this.removeEventListener("click", clickHandler);
    }, 150);
  };

  // Add the click event listener on the anchor element
  // Comment out this line if you don't want a one-off download of the blob content
  a.addEventListener("click", clickHandler, false);

  a.click();
};

let weightDivide = 5;

function setWeightDivision(value) {
  const scale = parseInt(value);
  d3.select("#link_WeightDividerOutput").text(scale);
  weightDivide = scale;
  setLinkWidth("weight");
}

function setLinkWidth(widthType) {
  link.attr("stroke-width", function (d) {
    const weightDivider = document.getElementById("weight-divider-label");
    if (widthType === "static") {
      weightDivider.style.display = "none";
      return 0.5;
    }
    if (widthType === "weight") {
      weightDivider.style.display = "block";
      return d.weight / weightDivide;
    }
    return 1;
  });
}