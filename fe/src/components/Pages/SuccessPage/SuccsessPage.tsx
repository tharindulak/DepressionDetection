import React, {useEffect} from 'react';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import styled from '@emotion/styled';
import { css, cx } from "@emotion/css";

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

export const SuccessTitleStyles = cx(css`
  font-weight: bold !important;
  height: 50px;
  margin-top: 10px !important;
  margin-left: 35px !important;
  color: #1877F2 !important;
`);

export const ErrorTitleStyles = cx(css`
  font-weight: bold !important;
  height: 50px;
  margin-top: 10px !important;
  margin-left: 35px !important;
  color: #f21839 !important;
`);


export const TextFieldStyles = cx(css`
  margin: 10px 10px 20px -118px !important;
  height: 40px;
  width: 500px;
`);

export const ButtonStyles = cx(css`
  margin-left: 10px !important;
`);

const SuccessPage = (props: {msg: string, error: boolean}) => {
    const {error, msg } = props;
    return (
        <Container>
            <Typography variant="h5" className={error ? ErrorTitleStyles : SuccessTitleStyles}>
                {msg}
            </Typography>
        </Container>
    );
};

export default SuccessPage;
