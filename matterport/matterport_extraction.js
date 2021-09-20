var iframe = document.getElementById('showcase');
var text = document.getElementById('text');
var button = document.getElementById('button');

// this key only works from jsfiddle.
var jsFiddleKey = "6eb9607db19546aebe10dce90aa001fa";
var sdkVersion = "3.7";
var modelSid = "";
// "aUKcUqeHV7k";
// "USLN9hG2mCB";
// "RSUX4QRxuGt";
// "fDjFC8xKFnV";
// "EBWtg8aRqv";

function pointToString(point) {
  var x = point.x.toFixed(3);
  var y = point.y.toFixed(3);
  var z = point.z.toFixed(3);

  return `{ x: ${x}, y: ${y}, z: ${z} }`;
}

function rotToString(point) {
  var x = point.x.toFixed(3);
  var y = point.y.toFixed(3);  

  return `{ x: ${x}, y: ${y}}`;
}

const resolution = {
  width: 1920,
  height: 1080
};

const visibility = {
  mattertags: false,
  sweeps: false    
};

iframe.src = `https://my.matterport.com/show?m=${modelSid}&hr=0&play=1&title=0&qs=1`;

var counter = 0;
var img_idx = 0;
var uuids = [];
var get_uuids = false;
var moveTo_uuid = false;

window.MP_SDK.connect(iframe, jsFiddleKey, sdkVersion).then(async function(theSdk) {
  var sdk = theSdk;
  console.log('SDK Connected!');
  /* var string = "hello world"
  var blob = new Blob([string], {type: "text/plain;charset=utf-8"});
  saveAs(blob, "info.txt");
   */
  /* download("hello world", "dlText.txt", "text/plain"); */

  var poseCache;
  
/*   sdk.Camera.setRotation({ x: 0, y: -20 }, { speed: 7 })
  .then(function() {
    // Camera rotation complete.
  })
  .catch(function(error) {
    // Camera rotation error.
  }); */
    
    sdk.Sweep.data.subscribe({
      onCollectionUpdated: function (collection) {
        /* console.log('the entire up-to-date collection', collection) ; */
        for (var key in collection) {
          // check if the property/key is defined in the object itself, not in parent
          if (collection.hasOwnProperty(key)) {           
            /* console.log(key, collection[key]); */
            uuids.push(key);
            // console.log('uuids: ', uuids);    				
            }
        } 
        }
  	});
    
    
  sdk.Camera.pose.subscribe(function(pose) {
    poseCache = pose;
    console.log(pointToString(poseCache.position));
    
  });
  
  
  var l = 0; // uuids.lengths-1;
  var rot_x = -30.0;
  var rot_y = -180.0;
  var rot_x_inc = 5.0;
  var rot_y_inc = 15.0;
    
	setInterval(() => {    
      var uuid;
    	uuid = uuids[l]; // .pop();
    	console.log("pop uuid: ", uuid);
      
      console.log("move to uuid: ", uuid);
      const rotation = { x: rot_x, y: rot_y };
      const transition = sdk.Sweep.Transition.INSTANT;
      const transitionTime = 200;
      sdk.Sweep.moveTo(uuid, {
        rotation: rotation,
        transition: transition,
        transitionTime: transitionTime,
      })
        .		then(function(sweepId){
        // Move successful.
        console.log('Arrived at sweep ' + sweepId);     
        sdk.Renderer.takeScreenShot(resolution, visibility)
      .then(function (screenShotUri) {        

        saveAs(new Blob([screenShotUri], {type:"text/plain;charset=utf-8"}), Number(img_idx) + "_rgb.txt");
        
        // console.log(pointToString(poseCache.position));
        console.log(rotToString(poseCache.rotation));
        
        var x = poseCache.position.x.toFixed(3)
        var y = poseCache.position.y.toFixed(3)
        var z = poseCache.position.z.toFixed(3)
        var yaw = poseCache.rotation.y.toFixed(3)
        var pitch = poseCache.rotation.x.toFixed(3)
        pose = x + ' ' + y + ' ' + z + ' ' + pitch + ' ' + yaw
        var blob = new Blob([pose], {type: "text/plain;charset=utf-8"});
  			saveAs(blob, Number(img_idx) + "_pose.txt");
        
        l = l+1;
        if(l == uuids.length){
          l = 0;
          rot_x = rot_x + rot_x_inc;
          if(rot_x > 30){
            rot_x = -30;
            rot_y = rot_y + rot_y_inc;
            if(rot_y > 180){
              return;
            }
          }
        }
        img_idx = img_idx + 1;
      });
              
      })
        .catch(function(error){
        // Error with moveTo command
      });
  }, 3000);

});

