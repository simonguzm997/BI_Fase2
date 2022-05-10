import React from "react";

function Result (props) {
    return (
        <div>
            {props.resultado === "0" ? 
                <div className="d-flex flex-column align-items-center">
                    <div className="d-flex justify-content-center mt-3">
                        <span><b>The patient is suitable for cancer clinical trials.</b></span>            
                    </div>
                </div> : props.resultado === "1" ?
                 <div className="d-flex flex-column align-items-center">
                    <div className="d-flex justify-content-center mt-3">
                        <span><b>The patient is not suitable for cancer clinical trials.</b></span> 
                    </div>
                </div> : 
                <div className="d-flex flex-column align-items-center">
                    <div className="d-flex justify-content-center mt-3">
                        <span className="text-center"><b>Fill and submit to see elegibility.</b></span> 
                    </div>
                </div>
            }
        </div>
    )
}

export default Result;