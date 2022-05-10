import axios from 'axios';
import Result from "./result";
import React, { useState } from "react";
import {Form, Button} from 'react-bootstrap';


function FormSection () {
    const [datos, setDatos] = useState({
        condition: '',
    })
    const [resultado, setResultado] = useState("-")

    const handleInputChange = (event) => {
        setDatos({
            ...datos,
            [event.target.id] : event.target.value,
        })
    }
    const sendData = (event) => {
        event.preventDefault();
        prediccionElegibilidad(datos);
    }

    async function prediccionElegibilidad(datos) {
        var url = "https://api-pipeline-bi.herokuapp.com/predict"
        const headers = {
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json',
        }
        const info = JSON.stringify({ "texts": [{"processed_condition": datos.condition}]});
        axios
            .post(url, info, {headers} )
            .then((resp)=> {
               console.log(resp)
               console.log(resp.data.Predict[1])
               setResultado(resp.data.Predict[1])
            })
            .catch((err)=> {
                console.log(err);
            }) 
    }
    return (
        <div className="row">
            <div className="col-12 border">
                <Form onSubmit={sendData}>
                    <Form.Group className="mb-3">
                        <Form.Label><b>Study and Condition</b></Form.Label>
                        <Form.Control type="text" placeholder="Enter text of the condition" id="condition" onChange={handleInputChange}/>
                    </Form.Group>
                    <div className='justify-content-center'>
                        <Button className="bg-success mt-3 mb-3" variant="primary" type="submit">
                            Submit
                        </Button>
                    </div>
                </Form>
            </div>
            <div className="col-12 border d-flex align-items-center justify-content-center">
                <Result resultado={resultado}/>
            </div>  
        </div> 
    )
}

export default FormSection;