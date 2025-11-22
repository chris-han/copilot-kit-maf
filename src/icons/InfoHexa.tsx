import { Info as LucideInfo, LucideProps } from 'lucide-react';

const InfoHexa = ({ className, ...props }: LucideProps) => {
  return <LucideInfo className={className} {...props} />;
};

export default InfoHexa;