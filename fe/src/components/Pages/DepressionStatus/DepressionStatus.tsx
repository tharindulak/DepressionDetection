import React from 'react';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import styled from '@emotion/styled';
import { css, cx } from "@emotion/css";
import { Button, Grid } from "@mui/material";
import axios from "axios";

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  justify-content: flex-start;
  padding-top: 20px;
  padding-left: 250px;
  background-color: #fff;
  border-radius: 5px;
`;

export const TitleStyles = cx(css`
  font-weight: bold !important;
  height: 50px;
  margin-top: 10px !important;
  margin-left: 35px !important;
  color: #1877F2 !important;
`);

export const SubTitleStyles = cx(css`
  height: 50px;
  margin-top: 10px !important;
  margin-left: 35px !important;
  color: #008355 !important;
`);

export const TextFieldStyles = cx(css`
  margin: 10px 10px 20px -25px !important;
  height: 40px;
  width: 500px;
`);

export const ButtonStyles = cx(css`
  margin-top: 20px !important;
  margin-left: 10px !important;
`);

const DepressionStatus = () => {
    const [id, setId] = React.useState(0);
    const [depressionValue, setDepressionValue] = React.useState(0);
    const [color, setColor] = React.useState(`rgb(255, 255, 255)`);

    function setColorFromDepressedValue(percentage: number) {
        // Convert the percentage to a value between 0 and 1
        const normalizedPercentage = percentage;

        // Calculate the RGB values for red and green
        const red = 255 * normalizedPercentage;
        const green = 255 * (1 - normalizedPercentage);

        // Create the RGB color string
        setColor(`rgb(${red}, ${green}, 0)`);
    }

    const handleSubmit = async () => {
        try {
            const response = await axios.get(`http://127.0.0.1:5000/satisfaction/${id}`);
            setDepressionValue(response.data.depressionStatus);
            setColorFromDepressedValue(response.data.depressionStatus/5);
        } catch (error) {
            console.error('Error sending JSON data:', error);
        }
    };

    const percentage = (depressionValue / 5) * 100;

    return (
        <Container>
            <Typography variant="h5" className={TitleStyles}>
                Depression Status
            </Typography>
            <Grid container spacing={2}>
                <Grid item xs={6}>
                    <TextField
                        value={id}
                        label="Employee Id"
                        variant="outlined"
                        className={TextFieldStyles}
                        onChange={(event: any) => { setId(event.target.value) }}
                    />
                    <Button className={ButtonStyles} onClick={handleSubmit} variant="contained">Submit</Button>
                </Grid>
            </Grid>
            <div>
                <Typography variant="h6" className={SubTitleStyles}>
                    Depression Score
                </Typography>
            </div>
            <div style={{display: "flex", marginLeft: 40, background: color}}>
                <svg width="200" height="100">
                    <rect x="0" y="0" width="200" height="100" stroke="#000" stroke-width="2px" fill={color}/>
                    <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle">{`${percentage}%`}</text>
                </svg>
            </div>
        </Container>
    );
};

export default DepressionStatus;
