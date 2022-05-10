import React from "react";
import FormSection from "./formSection";
import NavbarSection from "./navbarSection";

function Main () {
    return (
        <div>
            <NavbarSection />
            <div className="container mt-5 d-flex align-items-center">
                <h3>Determine the elegibility of a patient for cancer clinical trials</h3>
            </div>
            <div className="container mt-3">
                <div>
                    <FormSection />
                </div>
            </div>
        </div>
    );
}

export default Main;