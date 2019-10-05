
function upload() {
      var fileChooser = document.getElementById('file-chooser');
      var file = fileChooser.files[0];
      if (file) {            
          AWS.config.update({
              "accessKeyId": "AKIAJNZKPV4CVQYVVALQ",
              "secretAccessKey": "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",
              "region": "us-east-1"
          });
          var s3 = new AWS.S3();
          var params = {
              Bucket: 'cse6242-jkim3181',
              Key: file.name,
              ContentType: file.type,
              Body: file,
              ACL: 'public-read'
          };        
          s3.putObject(params, function (err, res) {
              if (err) {
                  console.log("Error uploading data: ", err);
              } else {
                  console.log("Successfully uploaded data");
                    var elem = document.getElementById("myBar");
                    document.getElementById("myProgress").style.display = "block";
                    var width = 10;
                    var id = setInterval(frame, 10);
                    function frame() {
                      if (width >= 100) {
                        clearInterval(id);
                        document.getElementById("analyse").disabled = false;
                      } else {
                        width++; 
                        elem.style.width = width + '%'; 
                        elem.innerHTML = width * 1  + '%';
                      }
                    }
              }
          });
      } else {
          alert('Nothing to upload.');
      }

      var totally_safe = 0;
      var pretty_safe = 0;
      var critical = 0;
      var moderate_critical = 0;
      var very_critical = 0;

      d3.csv("data/critical_patients_records.csv", function(data) {
        for (var i = 0; i < data.length; i++) {
          if (data[i].patient_category == "totally-safe") {
            totally_safe +=1;
          } else if(data[i].patient_category == "pretty-safe") {
            pretty_safe +=1;
          } else if(data[i].patient_category == "critical") {
            critical += 1;
          } else if(data[i].patient_category == "moderate-critical") {
            moderate_critical +=1;
          } else {
            very_critical +=1;
          }
        }
        console.log(totally_safe);
        console.log(pretty_safe);
        console.log(critical);
        console.log(moderate_critical);
        console.log(very_critical);
        //console.log(data.length);
      });
}