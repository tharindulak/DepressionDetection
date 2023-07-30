import React, {useEffect} from 'react';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import styled from '@emotion/styled';
import { css, cx } from "@emotion/css";
import { InputLabel, Select, Box, FormControl, Button, Grid } from "@mui/material";
import MenuItem from '@mui/material/MenuItem';
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

export const TextFieldStyles = cx(css`
  margin: 10px 10px 20px -118px !important;
  height: 40px;
  width: 500px;
`);

export const ButtonStyles = cx(css`
  margin-left: 10px !important;
`);

const SatisfactionData = () => {
    const [name, setName] = React.useState('');
    const [education, setEducation] = React.useState('');
    const [employeeCount, setEmployeeCount] = React.useState('1');
    const [employeeNumber, setEmployeeNumber] = React.useState('');
    const [environmentSatisfaction, setEnvironmentSatisfaction] = React.useState('');
    const [gender, setGender] = React.useState('');
    const [hourlyRate, setHourlyRate] = React.useState('');
    const [jobInvolvement, setJobInvolvement] = React.useState('');
    const [jobLevel, setJobLevel] = React.useState('');
    const [jobRole, setJobRole] = React.useState('');
    const [jobSatisfaction, setJobSatisfaction] = React.useState('');
    const [maritalStatus, setMaritalStatus] = React.useState('');
    const [monthlyIncome, setMonthlyIncome] = React.useState('');
    const [monthlyRate, setMonthlyRate] = React.useState('50000');
    const [numCompaniesWorked, setNumCompaniesWorked] = React.useState('');
    const [over18, setOver18] = React.useState('');
    const [overTime, setOverTime] = React.useState('');
    const [percentSalaryHike, setPercentSalaryHike] = React.useState('');
    const [performanceRating, setPerformanceRating] = React.useState('');
    const [relationshipSatisfaction, setRelationshipSatisfaction] = React.useState('');
    const [standardHours, setStandardHours] = React.useState('8');
    const [stockOptionLevel, setStockOptionLevel] = React.useState('');
    const [totalWorkingYears, setTotalWorkingYears] = React.useState('');
    const [trainingTimesLastYear, setTrainingTimesLastYear] = React.useState('');
    const [workLifeBalance, setWorkLifeBalance] = React.useState('');
    const [yearsAtCompany, setYearsAtCompany] = React.useState('');
    const [yearsInCurrentRole, setYearsInCurrentRole] = React.useState('');
    const [yearsSinceLastPromotion, setYearsSinceLastPromotion] = React.useState('');
    const [yearsWithCurrManager, setYearsWithCurrManager] = React.useState('');
    const [allFieldsFilled, setAllFieldsFilled] = React.useState(false);

    const validateNumberInput = (inputValue: any) => {
        return inputValue.replace(/\D/g, ""); // Remove any non-numeric characters
    };

    const checkAllFieldsFilled = () => {
        if (
            name &&
            education &&
            employeeCount &&
            employeeNumber &&
            environmentSatisfaction &&
            gender &&
            hourlyRate &&
            jobInvolvement &&
            jobLevel &&
            jobRole &&
            jobSatisfaction &&
            maritalStatus &&
            monthlyIncome &&
            monthlyRate &&
            numCompaniesWorked &&
            over18 &&
            overTime &&
            percentSalaryHike &&
            performanceRating &&
            relationshipSatisfaction &&
            standardHours &&
            stockOptionLevel &&
            totalWorkingYears &&
            trainingTimesLastYear &&
            workLifeBalance &&
            yearsAtCompany &&
            yearsInCurrentRole &&
            yearsSinceLastPromotion &&
            yearsWithCurrManager
        ) {
            setAllFieldsFilled(true);
        } else {
            setAllFieldsFilled(false);
        }
    };
    const handleClearFields = () => {
        setName("");
        setEducation("");
        setEmployeeCount("");
        setEmployeeNumber("");
        setEnvironmentSatisfaction("");
        setGender("");
        setHourlyRate("");
        setJobInvolvement("");
        setJobLevel("");
        setJobRole("");
        setJobSatisfaction("");
        setMaritalStatus("");
        setMonthlyIncome("");
        setMonthlyRate("");
        setNumCompaniesWorked("");
        setOver18("");
        setOverTime("");
        setPercentSalaryHike("");
        setPerformanceRating("");
        setRelationshipSatisfaction("");
        setStandardHours("");
        setStockOptionLevel("");
        setTotalWorkingYears("");
        setTrainingTimesLastYear("");
        setWorkLifeBalance("");
        setYearsAtCompany("");
        setYearsInCurrentRole("");
        setYearsSinceLastPromotion("");
        setYearsWithCurrManager("");
        setAllFieldsFilled(false); // Reset the state of allFieldsFilled to false after clearing the fields
    };

    const handleSubmit = async () => {
        const monthlyI = parseFloat(monthlyIncome) / 2;
        const jsonData = {
            name,
            education,
            employeeCount,
            employeeNumber,
            environmentSatisfaction,
            gender,
            hourlyRate,
            jobInvolvement,
            jobLevel,
            jobRole,
            jobSatisfaction,
            maritalStatus,
            monthlyI,
            monthlyRate,
            numCompaniesWorked,
            over18,
            overTime,
            percentSalaryHike,
            performanceRating,
            relationshipSatisfaction,
            standardHours,
            stockOptionLevel,
            totalWorkingYears,
            trainingTimesLastYear,
            workLifeBalance,
            yearsAtCompany,
            yearsInCurrentRole,
            yearsSinceLastPromotion,
            yearsWithCurrManager,
        };
        try {
            const response = await axios.post('http://127.0.0.1:5000/satisfaction', jsonData);
            console.log('Response from server:', response.data);
            // Do something with the response if needed
        } catch (error) {
            console.error('Error sending JSON data:', error);
        }
    };


    useEffect(() => {
        checkAllFieldsFilled();
    }, [
        name,
        education,
        employeeCount,
        employeeNumber,
        environmentSatisfaction,
        gender,
        hourlyRate,
        jobInvolvement,
        jobLevel,
        jobRole,
        jobSatisfaction,
        maritalStatus,
        monthlyIncome,
        monthlyRate,
        numCompaniesWorked,
        over18,
        overTime,
        percentSalaryHike,
        performanceRating,
        relationshipSatisfaction,
        standardHours,
        stockOptionLevel,
        totalWorkingYears,
        trainingTimesLastYear,
        workLifeBalance,
        yearsAtCompany,
        yearsInCurrentRole,
        yearsSinceLastPromotion,
        yearsWithCurrManager,
    ]);

    return (
        <Container>
            <Typography variant="h5" className={TitleStyles}>
                Employee Satisfaction Data
            </Typography>
            <Grid container spacing={2}>
                <Grid item xs={6}>
                    <TextField
                        value={name}
                        label="Name"
                        variant="outlined"
                        className={TextFieldStyles}
                        onChange={(event: any) => { setName(event.target.value) }}
                    />
                    <Box sx={{ width: 200, marginTop: "5px", marginLeft: "40px", marginBottom: "0px" }}>
                        <FormControl fullWidth>
                            <InputLabel id="education-label">Education</InputLabel>
                            <Select
                                labelId="education-label"
                                id="education"
                                value={education}
                                label="Education"
                                onChange={(event: any) => { setEducation(event.target.value) }}
                            >
                                <MenuItem value="Life Sciences">Life Sciences</MenuItem>
                                <MenuItem value="Medical">Medical</MenuItem>
                                <MenuItem value="Marketing">Marketing</MenuItem>
                                <MenuItem value="Other">Other</MenuItem>
                            </Select>
                        </FormControl>
                    </Box>
                    <TextField
                        label="Employee Number"
                        variant="outlined"
                        className={TextFieldStyles}
                        value={employeeNumber}
                        onChange={(event: any) => { setEmployeeNumber(validateNumberInput(event.target.value)) }}
                    />
                    <Box sx={{ width: 200, marginTop: "10px", marginLeft: "40px", marginBottom: "10px" }}>
                        <FormControl fullWidth>
                            <InputLabel id="environment-satisfaction-label">Environment Satisfaction</InputLabel>
                            <Select
                                labelId="environment-satisfaction-label"
                                id="environment-satisfaction"
                                value={environmentSatisfaction}
                                label="Environment Satisfaction"
                                onChange={(event: any) => { setEnvironmentSatisfaction(event.target.value) }}
                            >
                                <MenuItem value="1">1</MenuItem>
                                <MenuItem value="2">2</MenuItem>
                                <MenuItem value="3">3</MenuItem>
                                <MenuItem value="4">4</MenuItem>
                                <MenuItem value="5">5</MenuItem>
                            </Select>
                        </FormControl>
                    </Box>
                    <Box sx={{ width: 200, marginTop: "20px", marginLeft: "40px" }}>
                        <FormControl fullWidth>
                            <InputLabel id="gender-label">Gender</InputLabel>
                            <Select
                                labelId="gender-label"
                                id="gender"
                                value={gender}
                                label="Gender"
                                onChange={(event: any) => { setGender(event.target.value) }}
                            >
                                <MenuItem value="Male">Male</MenuItem>
                                <MenuItem value="Female">Female</MenuItem>
                            </Select>
                        </FormControl>
                    </Box>
                    <TextField
                        label="Hourly Rate"
                        variant="outlined"
                        className={TextFieldStyles}
                        value={hourlyRate}
                        onChange={(event: any) => { setHourlyRate(validateNumberInput(event.target.value)) }}
                    />
                    <Box sx={{ width: 200, marginTop: "10px", marginLeft: "40px" }}>
                        <FormControl fullWidth>
                            <InputLabel id="job-involvement-label">Job Involvement</InputLabel>
                            <Select
                                labelId="job-involvement-label"
                                id="job-involvement"
                                value={jobInvolvement}
                                label="Job Involvement"
                                onChange={(event: any) => { setJobInvolvement(event.target.value) }}
                            >
                                <MenuItem value="1">1</MenuItem>
                                <MenuItem value="2">2</MenuItem>
                                <MenuItem value="3">3</MenuItem>
                                <MenuItem value="4">4</MenuItem>
                                <MenuItem value="5">5</MenuItem>
                            </Select>
                        </FormControl>
                    </Box>
                    <Box sx={{ width: 200, marginTop: "10px", marginLeft: "40px", marginBottom: "10px" }}>
                        <FormControl fullWidth>
                            <InputLabel id="job-level-label">Job Level</InputLabel>
                            <Select
                                labelId="job-level-label"
                                id="job-level"
                                value={jobLevel}
                                label="Job Level"
                                onChange={(event: any) => { setJobLevel(event.target.value) }}
                            >
                                <MenuItem value="1">1</MenuItem>
                                <MenuItem value="2">2</MenuItem>
                                <MenuItem value="3">3</MenuItem>
                                <MenuItem value="4">4</MenuItem>
                                <MenuItem value="5">5</MenuItem>
                            </Select>
                        </FormControl>
                    </Box>
                    <Box sx={{ width: 200, marginTop: "10px", marginLeft: "40px", marginBottom: "10px" }}>
                        <FormControl fullWidth>
                            <InputLabel id="job-satisfaction-label">Job Role</InputLabel>
                            <Select
                                labelId="job-satisfaction-label"
                                id="job-role"
                                value={jobRole}
                                label="Job Satisfaction"
                                onChange={(event: any) => { setJobRole(event.target.value) }}
                            >
                                <MenuItem value="Sales Executive">Sales Executive</MenuItem>
                                <MenuItem value="Research Scientist">Research Scientist</MenuItem>
                                <MenuItem value="Laboratory Technician">Laboratory Technician</MenuItem>
                                <MenuItem value="Manufacturing Director">Manufacturing Director</MenuItem>
                                <MenuItem value="Healthcare Representative">Healthcare Representative</MenuItem>
                                <MenuItem value="Manager">Manager</MenuItem>
                                <MenuItem value="Manufacturing Director">Manufacturing Director</MenuItem>
                            </Select>
                        </FormControl>
                    </Box>
                    <Box sx={{ width: 200, marginTop: "10px", marginLeft: "40px", marginBottom: "10px" }}>
                        <FormControl fullWidth>
                            <InputLabel id="job-satisfaction-label">Job Satisfaction</InputLabel>
                            <Select
                                labelId="job-satisfaction-label"
                                id="job-satisfaction"
                                value={jobSatisfaction}
                                label="Job Satisfaction"
                                onChange={(event: any) => { setJobSatisfaction(event.target.value) }}
                            >
                                <MenuItem value="1">1</MenuItem>
                                <MenuItem value="2">2</MenuItem>
                                <MenuItem value="3">3</MenuItem>
                                <MenuItem value="4">4</MenuItem>
                                <MenuItem value="5">5</MenuItem>
                            </Select>
                        </FormControl>
                    </Box>
                    <Box sx={{ width: 200, marginTop: "20px", marginLeft: "40px" }}>
                        <FormControl fullWidth>
                            <InputLabel id="marital-status-label">Marital Status</InputLabel>
                            <Select
                                labelId="marital-status-label"
                                id="marital-status"
                                value={maritalStatus}
                                label="Marital Status"
                                onChange={(event: any) => { setMaritalStatus(event.target.value) }}
                            >
                                <MenuItem value="Single">Single</MenuItem>
                                <MenuItem value="Married">Married</MenuItem>
                                <MenuItem value="Divorced">Divorced</MenuItem>
                            </Select>
                        </FormControl>
                    </Box>
                    <TextField
                        label="Monthly Income"
                        variant="outlined"
                        className={TextFieldStyles}
                        value={monthlyIncome}
                        onChange={(event: any) => { setMonthlyIncome(validateNumberInput(event.target.value)) }}
                    />
                    <TextField
                        label="Number of Companies Worked"
                        variant="outlined"
                        className={TextFieldStyles}
                        value={numCompaniesWorked}
                        onChange={(event: any) => { setNumCompaniesWorked(validateNumberInput(event.target.value)) }}
                    />
                </Grid>
                <Grid item xs={6}>
                    <Box sx={{ width: 200, marginTop: "10px", marginLeft: "40px", marginBottom: "10px" }}>
                        <FormControl fullWidth>
                            <InputLabel id="over18-label">Over 18</InputLabel>
                            <Select
                                labelId="over18-label"
                                id="over18"
                                value={over18}
                                label="Over 18"
                                onChange={(event: any) => { setOver18(event.target.value) }}
                            >
                                <MenuItem value="Y">Yes</MenuItem>
                                <MenuItem value="N">No</MenuItem>
                            </Select>
                        </FormControl>
                    </Box>
                    <Box sx={{ width: 200, marginTop: "20px", marginLeft: "40px" }}>
                        <FormControl fullWidth>
                            <InputLabel id="overtime-label">Over Time</InputLabel>
                            <Select
                                labelId="overtime-label"
                                id="overtime"
                                value={overTime}
                                label="Over Time"
                                onChange={(event: any) => { setOverTime(event.target.value) }}
                            >
                                <MenuItem value="Yes">Yes</MenuItem>
                                <MenuItem value="No">No</MenuItem>
                            </Select>
                        </FormControl>
                    </Box>
                    <TextField
                        label="Percent Salary Hike"
                        variant="outlined"
                        className={TextFieldStyles}
                        value={percentSalaryHike}
                        onChange={(event: any) => { setPercentSalaryHike(validateNumberInput(event.target.value))}}
                    />
                    <Box sx={{ width: 200, marginTop: "10px", marginLeft: "40px", marginBottom: "10px" }}>
                        <FormControl fullWidth>
                            <InputLabel id="performance-rating-label">Performance Rating</InputLabel>
                            <Select
                                labelId="performance-rating-label"
                                id="performance-rating"
                                value={performanceRating}
                                label="Performance Rating"
                                onChange={(event: any) => { setPerformanceRating(event.target.value) }}
                            >
                                <MenuItem value="1">1</MenuItem>
                                <MenuItem value="2">2</MenuItem>
                                <MenuItem value="3">3</MenuItem>
                                <MenuItem value="4">4</MenuItem>
                                <MenuItem value="5">5</MenuItem>
                            </Select>
                        </FormControl>
                    </Box>
                    <Box sx={{ width: 200, marginTop: "10px", marginLeft: "40px", marginBottom: "0px" }}>
                        <FormControl fullWidth>
                            <InputLabel id="relationship-satisfaction-label">Relationship Satisfaction</InputLabel>
                            <Select
                                labelId="relationship-satisfaction-label"
                                id="relationship-satisfaction"
                                value={relationshipSatisfaction}
                                label="Relationship Satisfaction"
                                onChange={(event: any) => { setRelationshipSatisfaction(event.target.value) }}
                            >
                                <MenuItem value="1">1</MenuItem>
                                <MenuItem value="2">2</MenuItem>
                                <MenuItem value="3">3</MenuItem>
                                <MenuItem value="4">4</MenuItem>
                                <MenuItem value="5">5</MenuItem>
                            </Select>
                        </FormControl>
                    </Box>
                    <Box sx={{ width: 200, marginTop: "10px", marginLeft: "40px", marginBottom: "10px" }}>
                        <FormControl fullWidth>
                            <InputLabel id="stock-option-level-label">Stock Option Level</InputLabel>
                            <Select
                                labelId="stock-option-level-label"
                                id="stock-option-level"
                                value={stockOptionLevel}
                                label="Stock Option Level"
                                onChange={(event: any) => { setStockOptionLevel(event.target.value) }}
                            >
                                <MenuItem value="1">1</MenuItem>
                                <MenuItem value="2">2</MenuItem>
                                <MenuItem value="3">3</MenuItem>
                                <MenuItem value="4">4</MenuItem>
                                <MenuItem value="5">5</MenuItem>
                            </Select>
                        </FormControl>
                    </Box>
                    <TextField
                        label="Total Working Years"
                        variant="outlined"
                        className={TextFieldStyles}
                        value={totalWorkingYears}
                        onChange={(event: any) => { setTotalWorkingYears(validateNumberInput(event.target.value)) }}
                    />
                    <TextField
                        label="Training Times Last Year"
                        variant="outlined"
                        className={TextFieldStyles}
                        value={trainingTimesLastYear}
                        onChange={(event: any) => { setTrainingTimesLastYear(validateNumberInput(event.target.value))}}
                    />
                    <Box sx={{ width: 200, marginTop: "10px", marginLeft: "40px", marginBottom: "10px" }}>
                        <FormControl fullWidth>
                            <InputLabel id="work-life-balance-label">Work Life Balance</InputLabel>
                            <Select
                                labelId="work-life-balance-label"
                                id="work-life-balance"
                                value={workLifeBalance}
                                label="Work Life Balance"
                                onChange={(event: any) => { setWorkLifeBalance(event.target.value) }}
                            >
                                <MenuItem value="1">1</MenuItem>
                                <MenuItem value="2">2</MenuItem>
                                <MenuItem value="3">3</MenuItem>
                                <MenuItem value="4">4</MenuItem>
                                <MenuItem value="5">5</MenuItem>
                            </Select>
                        </FormControl>
                    </Box>
                    <TextField
                        label="Years at Company"
                        variant="outlined"
                        className={TextFieldStyles}
                        value={yearsAtCompany}
                        onChange={(event: any) => { setYearsAtCompany(validateNumberInput(event.target.value)) }}
                    />
                    <TextField
                        label="Years in Current Role"
                        variant="outlined"
                        className={TextFieldStyles}
                        value={yearsInCurrentRole}
                        onChange={(event: any) => { setYearsInCurrentRole(validateNumberInput(event.target.value)) }}
                    />
                    <Box sx={{ width: 200, marginTop: "10px", marginLeft: "40px", marginBottom: "10px" }}>
                        <FormControl fullWidth>
                            <InputLabel id="years-since-last-promotion-label">Years Since Last Promotion</InputLabel>
                            <Select
                                labelId="years-since-last-promotion-label"
                                id="years-since-last-promotion"
                                value={yearsSinceLastPromotion}
                                label="Years Since Last Promotion"
                                onChange={(event: any) => { setYearsSinceLastPromotion(event.target.value) }}
                            >
                                <MenuItem value="1">1</MenuItem>
                                <MenuItem value="2">2</MenuItem>
                                <MenuItem value="3">3</MenuItem>
                                <MenuItem value="4">4</MenuItem>
                                <MenuItem value="5">5</MenuItem>
                                <MenuItem value="6">6</MenuItem>
                                <MenuItem value="7">7</MenuItem>
                                <MenuItem value="8">8</MenuItem>
                                <MenuItem value="9">9</MenuItem>
                                <MenuItem value="10">10</MenuItem>
                            </Select>
                        </FormControl>
                    </Box>
                    <TextField
                        label="Years With Current Manager"
                        variant="outlined"
                        className={TextFieldStyles}
                        value={yearsWithCurrManager}
                        onChange={(event: any) => { setYearsWithCurrManager(validateNumberInput(event.target.value)) }}
                    />
                </Grid>
            </Grid>
            <div style={{display: "flex", flexDirection: "row", marginLeft: 1100, marginBottom: 20, marginTop: 10}}>
                <Button className={ButtonStyles} variant="outlined" onClick={handleClearFields}>Clear</Button>
                <Button className={ButtonStyles} disabled={!allFieldsFilled} onClick={handleSubmit} variant="contained">Submit</Button>
            </div>
        </Container>
    );
};

export default SatisfactionData;
