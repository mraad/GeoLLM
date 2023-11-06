import Map from "@arcgis/core/Map.js";
import GeoJSONLayer from "@arcgis/core/layers/GeoJSONLayer.js";
import MapView from "@arcgis/core/views/MapView.js";
import "./style.css";

// If GeoJSON files are not on the same domain as your website, a CORS enabled server
// or a proxy is required.
const url_points = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson";
const url_polygons = "http://localhost:8000/output_data/geojson_1693523078.json"

//
// Earthquake Analysis
//
// Paste the url into a browser's address bar to download and view the attributes
// in the GeoJSON file. These attributes include:
// * mag - magnitude
// * type - earthquake or other event such as nuclear test
// * place - location of the event
// * time - the time of the event
// Use the Arcade Date() function to format time field into a human-readable format

const template_points = {
  title: "Earthquake Info",
  content: "Magnitude {mag} {type} hit {place} on {time}",
  fieldInfos: [
    {
      fieldName: "time",
      format: {
        dateFormat: "short-date-short-time"
      }
    }
  ]
};

const renderer_points = {
  type: "simple",
  field: "mag",
  symbol: {
    type: "simple-marker",
    color: "orange",
    outline: {
      color: "white"
    }
  },
  visualVariables: [
    {
      type: "size",
      field: "mag",
      stops: [
        {
          value: 2.5,
          size: "4px"
        },
        {
          value: 8,
          size: "40px"
        }
      ]
    }
  ]
};

const geojsonLayer_points = new GeoJSONLayer({
  url: url_points,
  copyright: "USGS Earthquakes",
  popupTemplate: template_points,
  renderer: renderer_points,
  orderBy: {
    field: "mag"
  }
});

//
// Chicago Crime Data Analysis
//
const template_polygons = {
  title: "Chicago Crime Info",
  content: "Count {count}",
  fieldInfos: [
    {
      fieldName: "count",
    }
  ]
};

const less5 = {
  type: "simple-fill", // autocasts as new SimpleFillSymbol()
  color: "#fffcd4",
  style: "solid",
  outline: {
    width: 0.2,
    color: [255, 255, 255, 0.5]
  }
};

const less10 = {
  type: "simple-fill", // autocasts as new SimpleFillSymbol()
  color: "#b1cdc2",
  style: "solid",
  outline: {
    width: 0.2,
    color: [255, 255, 255, 0.5]
  }
};

const less20 = {
  type: "simple-fill", // autocasts as new SimpleFillSymbol()
  color: "#38627a",
  style: "solid",
  outline: {
    width: 0.2,
    color: [255, 255, 255, 0.5]
  }
};

const more20 = {
  type: "simple-fill", // autocasts as new SimpleFillSymbol()
  color: "#0d2644",
  style: "solid",
  outline: {
    width: 0.2,
    color: [255, 255, 255, 0.5]
  }
};

const renderer_polygons = {
  type: "class-breaks", // autocasts as new ClassBreaksRenderer()
  field: "count", // total number of adults (25+) with a college degree
  defaultSymbol: {
    type: "simple-fill", // autocasts as new SimpleFillSymbol()
    color: "orange",
    style: "solid",
    outline: {
      width: 0.5,
      color: [50, 50, 50, 0.6]
    }
  },
  classBreakInfos: [
  {
    minValue: 0,
    maxValue: 5,
    symbol: less5,
    label: "< 5" // label for symbol in legend
  },
  {
    minValue: 5,
    maxValue: 10,
    symbol: less10,
    label: "5 - 10" // label for symbol in legend
  },
  {
    minValue: 10,
    maxValue: 20,
    symbol: less20,
    label: "10 - 20" // label for symbol in legend
  },
  {
    minValue: 20,
    maxValue: 70,
    symbol: more20,
    label: "> 20" // label for symbol in legend
  }
],
  defaultLabel: "no data" // legend label for features that don't match a class break
};

const geojsonLayer_polygons = new GeoJSONLayer({
  url: url_polygons,
  copyright: "Chicago Crime Data",
  popupTemplate: template_polygons,
  renderer: renderer_polygons,
  orderBy: {
    field: "mag"
  }
});

//
// Map
//
const map = new Map({
  basemap: "gray-vector",
  // layers: [geojsonLayer_points]
  layers: [geojsonLayer_polygons]
});

const view = new MapView({
  container: "viewDiv",
  center: [-168, 46],
  zoom: 2,
  map: map
});
